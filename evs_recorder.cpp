// ============================================================================
//  evs_recorder.cpp   (renamed from evs_recorder_mv.cpp -- see note below)
//  Long-duration SPARSE event recorder for LUCID Triton2 EVS (TRT009S-EC)
//  via Metavision SDK / OpenEB 4.6.2 + LUCID's HAL plugin.
//
//  RENAME NOTE: this file used to be named evs_recorder_mv.cpp, living
//  alongside a DIFFERENT evs_recorder.cpp (Arena SDK, DENSE accumulated
//  frames, .cevt). That Arena-based file is now RETIRED to legacy/
//  (legacy/evs_recorder.cpp) because Arena SDK cannot deliver sparse events
//  on this camera at all (see below) -- this file, the Metavision one, is
//  now the ONLY active recorder and took over the plain "evs_recorder.cpp"
//  name. build_linux.sh (renamed from build_mv.sh) builds this file into a
//  binary named `evs_recorder` (not `evs_recorder_mv`) by default now.
//
//    evs_recorder.cpp (THIS FILE)     Metavision -> SPARSE (x,y,p,t) events (.raw)
//    legacy/evs_recorder.cpp (RETIRED) Arena SDK  -> DENSE accumulated frames (.cevt)
//
//  WHY THIS FILE EXISTS
//  --------------------
//  On this camera, Arena SDK cannot deliver sparse events. That is a
//  firmware/GenICam limit, not a code bug, and it was established the hard
//  way: AcquisitionAccumulationMode (TimeBased/EventBased) is present on the
//  device node map but reports IsAvailable=false / IsReadable=false /
//  IsWritable=false; over 2160 nodes were swept for an unlock path and none
//  exists. Every Arena buffer is a dense width*height accumulated frame, so
//  sub-window event ordering is destroyed inside the camera and no honest
//  per-event microsecond timestamp can be recovered offline.
//
//  Metavision HAL bypasses that abstraction entirely and was CONFIRMED
//  working on this machine by probe_metavision.cpp, which printed 500 real
//  sparse events with microsecond timestamps. This recorder is the
//  production version of that probe.
//
//  WHAT THIS RECORDER WRITES
//  -------------------------
//  A standard Prophesee .raw file, written by the SDK itself via
//  Camera::start_recording(). Deliberately NOT a custom container this time:
//  the SDK's writer runs on its own thread (documented in camera.h as the
//  recommended way to save a recording, explicitly noted as not slowing the
//  decoding thread, unlike I_EventsStream::log_raw_data), and .raw is the
//  format the rest of the event-vision ecosystem already reads. That removes
//  a whole class of self-inflicted format bugs and removes the need for a
//  cevt_to_events.py equivalent on this path.
//
//  The CD callback registered below is for LIVE STATISTICS ONLY (event
//  count / rate / polarity split). It is deliberately not the write path.
//  Keeping it arithmetic-only means it cannot become a bottleneck that
//  causes the SDK to drop events.
//
//  A sidecar <output>.meta.json is written next to the .raw with camera
//  serial, plugin, geometry, bias values actually in effect, ERC state, and
//  host wall-clock start/stop — everything needed later to reproduce or
//  interpret the recording without guesswork.
//
//  API VERIFICATION
//  ----------------
//  Every Metavision call below was checked against the OpenEB 4.6.2 headers
//  (the exact version of the .so files LUCID bundles in
//  ArenaSDK_Linux_x64/Metavision/lib, confirmed via `strings`):
//    Camera::from_first_available() / from_serial()  camera.h:181,239
//    Camera::cd().add_callback(EventsCDCallback)     cd.h:44
//    Camera::start()/stop()/is_running()             camera.h:374
//    Camera::start_recording()/stop_recording()      camera.h:402,412
//    Camera::biases() -> Biases&                     camera.h:342
//    Biases::set_from_file()/save_to_file()          biases.h:34,38
//    Biases::get_facility() -> I_LL_Biases*          biases.h:41
//    I_LL_Biases::get_all_biases()                   i_ll_biases.h:53
//    Camera::get_device() -> Device&                 camera.h:450
//    Device::get_facility<T>() -> T* (may be null)   device.h:31
//    I_ErcModule::enable/is_enabled/set_cd_event_rate i_erc_module.h:30,34,43
//    I_Geometry::get_width()/get_height()            i_geometry.h:24,28
//    I_HW_Identification::get_serial()               i_hw_identification.h:69
//    EventCD fields x,y,p,t (unsigned short/short/timestamp) event2d.h:33-49
//
//  KNOWN SDK TEARDOWN BUG — WHY THIS EXITS VIA _Exit()
//  ---------------------------------------------------
//  probe_metavision.cpp completed successfully (all 500 events printed, clean
//  "Capture complete") and THEN aborted with:
//      terminate called after throwing an instance of
//      'boost::wrapexcept<boost::lock_error>'
//      what(): boost: mutex lock failed in pthread_mutex_lock: Invalid argument
//  That fires during static/global destruction inside the bundled Metavision
//  libraries, after all user work is finished. It does not affect captured
//  data. But a recorder that core-dumps on every exit is unusable in
//  unattended long runs (it pollutes logs and makes the exit code lie about
//  whether the recording succeeded), so once the .raw is closed and the
//  sidecar is flushed, this program leaves via std::_Exit(0), which skips
//  those destructors by design. See ExitNow().
//
//  BUILD
//    bash build_linux.sh         (renamed from build_mv.sh; see companion script)
//
//  TYPICAL USE
//    ./evs_recorder --output run01.raw --duration 60
//    ./evs_recorder --output run02.raw --bias-file night.bias --erc-rate 40000000
//    ./evs_recorder --info
//    ./evs_recorder --output x.raw --save-bias current.bias --duration 5
// ============================================================================

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/camera_exception.h>
#include <metavision/hal/facilities/i_erc_module.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/facilities/i_ll_biases.h>

// ---------------------------------------------------------------------------
// SECTION 0 — signal handling
//
// Async-signal-safe rule: a handler may only touch a volatile sig_atomic_t /
// lock-free atomic. No printing, no camera calls, no allocation in here. The
// main loop polls this flag and does the actual shutdown.
// ---------------------------------------------------------------------------
static std::atomic<bool> g_stopRequested{false};

extern "C" void HandleSignal(int)
{
	g_stopRequested.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// SECTION 1 — live statistics
//
// Updated from the SDK's decoding thread inside the CD callback, read from
// the main thread. relaxed ordering is correct here: these are monotonic
// counters used for human-readable progress, never for control flow that
// needs to observe a consistent snapshot across all four values.
// ---------------------------------------------------------------------------
struct Stats {
	std::atomic<uint64_t> totalEvents{0};
	std::atomic<uint64_t> positiveEvents{0};
	std::atomic<uint64_t> negativeEvents{0};
	std::atomic<int64_t>  lastEventTimestampUs{0}; // sensor clock, microseconds
};

static Stats g_stats;

// ---------------------------------------------------------------------------
// SECTION 2 — CLI
// ---------------------------------------------------------------------------
struct Args {
	std::string outputPath;
	std::string biasFileIn;    // --bias-file  : load before start
	std::string biasFileOut;   // --save-bias  : dump effective biases to file
	std::string serial;        // --serial     : pick a specific camera
	double      durationSec   = 0.0;   // 0 = until Ctrl-C
	double      statsInterval = 2.0;
	uint32_t    ercRate       = 0;     // events/sec; 0 = leave untouched
	bool        ercOff        = false;
	bool        infoOnly      = false;
	bool        showHelp      = false;
};

static void PrintUsage(const char *argv0)
{
	std::cout <<
		"Sparse event recorder for LUCID Triton2 EVS via Metavision SDK.\n"
		"Writes a standard Prophesee .raw file plus a .meta.json sidecar.\n"
		"\n"
		"Usage:\n"
		"  " << argv0 << " --output <path.raw> [options]\n"
		"  " << argv0 << " --info\n"
		"\n"
		"Options:\n"
		"  --output <path.raw>     Output file. Overwritten if it exists.\n"
		"  --duration <seconds>    Stop automatically after N seconds.\n"
		"                          Omit or 0 to record until Ctrl-C.\n"
		"  --bias-file <path>      Load a bias profile before starting. This is\n"
		"                          the Metavision replacement for Arena's\n"
		"                          UserSet1/UserSet2, which allowed only two\n"
		"                          on-camera slots. Here you can keep one file\n"
		"                          per lighting condition, unlimited.\n"
		"  --save-bias <path>      Write the biases actually in effect to a file\n"
		"                          (after --bias-file is applied, if given).\n"
		"                          Use this to snapshot a tuned configuration.\n"
		"  --erc-rate <ev/s>       Enable Event Rate Control at this target, in\n"
		"                          EVENTS PER SECOND (not Mev/s). LUCID recommends\n"
		"                          40 Mev/s on a 1GigE link -> --erc-rate 40000000.\n"
		"                          Note ERC DROPS events to stay under the target:\n"
		"                          leave it off for noise-floor/calibration work,\n"
		"                          turn it on to protect a long capture from\n"
		"                          link overflow.\n"
		"  --erc-off               Explicitly disable ERC before recording.\n"
		"  --serial <serial>       Open a specific camera instead of the first\n"
		"                          available one.\n"
		"  --stats-interval <sec>  Live stats print period (default 2.0, 0=off).\n"
		"  --info                  Print camera info, geometry, biases, ERC state,\n"
		"                          then exit without recording.\n"
		"  --help                  This message.\n"
		"\n"
		"Before running: close ArenaView / evs_recorder / anything Arena-based.\n"
		"Only one process can hold the GigE Vision control channel at a time.\n"
		"\n"
		"Environment (set these or the camera will not be found):\n"
		"  MV_HAL_PLUGIN_PATH  must include LUCID's hal_plugin directory\n"
		"  LD_LIBRARY_PATH     must include ArenaSDK_Linux_x64/Metavision/lib\n";
}

// Parses "--flag value" pairs. Returns false with a message on malformed input
// rather than silently continuing with a default, because a recorder that
// silently ignores "--duration 600" and then stops at Ctrl-C is worse than one
// that refuses to start.
static bool ParseArgs(int argc, char **argv, Args &out, std::string &error)
{
	auto needValue = [&](int &i, const char *flag, std::string &dst) -> bool {
		if (i + 1 >= argc) {
			error = std::string(flag) + " requires a value";
			return false;
		}
		dst = argv[++i];
		return true;
	};

	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		std::string v;

		if (a == "--help" || a == "-h") {
			out.showHelp = true;
		} else if (a == "--info") {
			out.infoOnly = true;
		} else if (a == "--erc-off") {
			out.ercOff = true;
		} else if (a == "--output") {
			if (!needValue(i, "--output", out.outputPath)) return false;
		} else if (a == "--bias-file") {
			if (!needValue(i, "--bias-file", out.biasFileIn)) return false;
		} else if (a == "--save-bias") {
			if (!needValue(i, "--save-bias", out.biasFileOut)) return false;
		} else if (a == "--serial") {
			if (!needValue(i, "--serial", out.serial)) return false;
		} else if (a == "--duration") {
			if (!needValue(i, "--duration", v)) return false;
			try { out.durationSec = std::stod(v); }
			catch (...) { error = "--duration expects a number of seconds"; return false; }
			if (out.durationSec < 0) { error = "--duration cannot be negative"; return false; }
		} else if (a == "--stats-interval") {
			if (!needValue(i, "--stats-interval", v)) return false;
			try { out.statsInterval = std::stod(v); }
			catch (...) { error = "--stats-interval expects a number of seconds"; return false; }
			if (out.statsInterval < 0) { error = "--stats-interval cannot be negative"; return false; }
		} else if (a == "--erc-rate") {
			if (!needValue(i, "--erc-rate", v)) return false;
			try { out.ercRate = static_cast<uint32_t>(std::stoul(v)); }
			catch (...) { error = "--erc-rate expects a positive integer (events/sec)"; return false; }
			if (out.ercRate == 0) { error = "--erc-rate must be > 0 (use --erc-off to disable ERC)"; return false; }
		} else {
			error = "unknown argument: " + a;
			return false;
		}
	}
	return true;
}

// ---------------------------------------------------------------------------
// SECTION 3 — small helpers
// ---------------------------------------------------------------------------

// ISO-8601-ish UTC stamp for the sidecar. Wall clock, used only for humans
// correlating this recording with an RGB capture or a lab notebook. All
// EVENT timing lives in the .raw and comes from the sensor clock.
static std::string UtcTimestampString()
{
	const auto now  = std::chrono::system_clock::now();
	const auto secs = std::chrono::system_clock::to_time_t(now);
	std::tm tm{};
	gmtime_r(&secs, &tm);
	char buf[32];
	std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
	return buf;
}

// Minimal JSON string escaping. Camera serials/plugin names are tame, but a
// sidecar that silently produces invalid JSON when a field contains a quote
// or backslash would break every downstream parser at the worst moment.
static std::string JsonEscape(const std::string &s)
{
	std::string out;
	out.reserve(s.size() + 8);
	for (const char c : s) {
		switch (c) {
			case '"':  out += "\\\""; break;
			case '\\': out += "\\\\"; break;
			case '\n': out += "\\n";  break;
			case '\r': out += "\\r";  break;
			case '\t': out += "\\t";  break;
			default:
				if (static_cast<unsigned char>(c) < 0x20) {
					char esc[7];
					std::snprintf(esc, sizeof(esc), "\\u%04x", c);
					out += esc;
				} else {
					out += c;
				}
		}
	}
	return out;
}

// Deliberate hard exit. See the "KNOWN SDK TEARDOWN BUG" note in the file
// header: the bundled Metavision libraries throw boost::lock_error during
// static destruction after all work is done. Everything this program owns
// (the .raw via stop_recording, the sidecar via an explicit flush+close) is
// already durable by the time this is called.
[[noreturn]] static void ExitNow(int code)
{
	std::cout.flush();
	std::cerr.flush();
	std::_Exit(code);
}

// ---------------------------------------------------------------------------
// SECTION 4 — camera introspection
//
// Facilities are optional by contract: Device::get_facility<T>() returns a
// raw pointer that may be null when a plugin does not implement T. Every
// access below is null-checked; a missing ERC or geometry facility must
// degrade to "unknown", never crash a recording.
// ---------------------------------------------------------------------------
struct CameraInfo {
	std::string serial;
	std::string plugin;
	std::string integrator;
	std::string encodingFormat;
	std::string firmware;
	std::string systemId;
	int         width  = 0;
	int         height = 0;
	bool        ercAvailable = false;
	bool        ercEnabled   = false;
	std::map<std::string, int> biases;
};

static CameraInfo CollectCameraInfo(Metavision::Camera &cam)
{
	CameraInfo info;

	const Metavision::CameraConfiguration &cfg = cam.get_camera_configuration();
	info.serial         = cfg.serial_number;
	info.plugin         = cfg.plugin_name;
	info.integrator     = cfg.integrator;
	info.encodingFormat = cfg.data_encoding_format;
	info.firmware       = cfg.firmware_version;
	info.systemId       = cfg.system_ID;

	Metavision::Device &dev = cam.get_device();

	if (const Metavision::I_Geometry *geo = dev.get_facility<Metavision::I_Geometry>()) {
		info.width  = geo->get_width();
		info.height = geo->get_height();
	}

	if (const Metavision::I_ErcModule *erc = dev.get_facility<Metavision::I_ErcModule>()) {
		info.ercAvailable = true;
		info.ercEnabled   = erc->is_enabled();
	}

	// Biases are read through the driver-level wrapper, which exposes the
	// underlying HAL facility. Null when a plugin has no low-level bias
	// control, so this is guarded too.
	if (const Metavision::I_LL_Biases *lb = cam.biases().get_facility()) {
		info.biases = lb->get_all_biases();
	}

	return info;
}

static void PrintCameraInfo(const CameraInfo &info)
{
	std::cout << "[camera] serial          = " << info.serial << "\n"
	          << "[camera] system ID       = " << info.systemId << "\n"
	          << "[camera] plugin          = " << info.plugin << "\n"
	          << "[camera] integrator      = " << info.integrator << "\n"
	          << "[camera] firmware        = " << info.firmware << "\n"
	          << "[camera] encoding format = " << info.encodingFormat << "\n";

	if (info.width > 0 && info.height > 0) {
		std::cout << "[camera] geometry        = " << info.width << "x" << info.height << "\n";
	} else {
		std::cout << "[camera] geometry        = unknown (no I_Geometry facility)\n";
	}

	if (info.ercAvailable) {
		std::cout << "[camera] ERC             = " << (info.ercEnabled ? "enabled" : "disabled") << "\n";
	} else {
		std::cout << "[camera] ERC             = not available on this plugin\n";
	}

	if (info.biases.empty()) {
		std::cout << "[camera] biases          = unavailable\n";
	} else {
		std::cout << "[camera] biases:\n";
		for (const auto &kv : info.biases) {
			std::cout << "           " << std::left << std::setw(28) << kv.first
			          << " = " << kv.second << "\n";
		}
	}
}

// ---------------------------------------------------------------------------
// SECTION 5 — sidecar metadata
//
// Written AFTER the run so it can carry the final counters. Explicitly
// flushed and closed before the process exits, since ExitNow() skips normal
// stream destruction.
// ---------------------------------------------------------------------------
static bool WriteSidecar(const std::string &rawPath,
                         const CameraInfo  &info,
                         const Args        &args,
                         const std::string &startedUtc,
                         const std::string &stoppedUtc,
                         double             wallSeconds,
                         const std::string &stopReason,
                         std::string       &errorOut)
{
	const std::string path = rawPath + ".meta.json";
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f) {
		errorOut = "could not open " + path + " for writing";
		return false;
	}

	const uint64_t total = g_stats.totalEvents.load(std::memory_order_relaxed);
	const uint64_t pos   = g_stats.positiveEvents.load(std::memory_order_relaxed);
	const uint64_t neg   = g_stats.negativeEvents.load(std::memory_order_relaxed);
	const int64_t  lastT = g_stats.lastEventTimestampUs.load(std::memory_order_relaxed);

	f << "{\n";
	f << "  \"raw_file\": \""            << JsonEscape(rawPath)          << "\",\n";
	f << "  \"recorder\": \"evs_recorder (Metavision SDK / OpenEB 4.6.2)\",\n";
	f << "  \"camera\": {\n";
	f << "    \"serial\": \""            << JsonEscape(info.serial)         << "\",\n";
	f << "    \"system_id\": \""         << JsonEscape(info.systemId)       << "\",\n";
	f << "    \"plugin\": \""            << JsonEscape(info.plugin)         << "\",\n";
	f << "    \"integrator\": \""        << JsonEscape(info.integrator)     << "\",\n";
	f << "    \"firmware\": \""          << JsonEscape(info.firmware)       << "\",\n";
	f << "    \"encoding_format\": \""   << JsonEscape(info.encodingFormat) << "\",\n";
	f << "    \"width\": "               << info.width                      << ",\n";
	f << "    \"height\": "              << info.height                     << "\n";
	f << "  },\n";

	f << "  \"erc\": {\n";
	f << "    \"available\": " << (info.ercAvailable ? "true" : "false") << ",\n";
	f << "    \"enabled\": "   << (info.ercEnabled   ? "true" : "false") << ",\n";
	if (args.ercRate > 0) {
		f << "    \"requested_rate_events_per_sec\": " << args.ercRate << "\n";
	} else {
		f << "    \"requested_rate_events_per_sec\": null\n";
	}
	f << "  },\n";

	f << "  \"biases\": {";
	bool first = true;
	for (const auto &kv : info.biases) {
		f << (first ? "\n" : ",\n") << "    \"" << JsonEscape(kv.first) << "\": " << kv.second;
		first = false;
	}
	f << (first ? "" : "\n") << "  },\n";
	f << "  \"bias_file_loaded\": "
	  << (args.biasFileIn.empty() ? "null" : ("\"" + JsonEscape(args.biasFileIn) + "\"")) << ",\n";

	f << "  \"run\": {\n";
	f << "    \"started_utc\": \""   << startedUtc  << "\",\n";
	f << "    \"stopped_utc\": \""   << stoppedUtc  << "\",\n";
	f << "    \"wall_seconds\": "    << std::fixed << std::setprecision(3) << wallSeconds << ",\n";
	f << "    \"stop_reason\": \""   << JsonEscape(stopReason) << "\"\n";
	f << "  },\n";

	// These counters come from the stats callback, which sees the DECODED
	// stream. Treat them as a close approximation of what landed in the .raw,
	// not as an authoritative count: the SDK's writer and the decoder are
	// separate consumers. The .raw itself is the source of truth.
	f << "  \"observed_events\": {\n";
	f << "    \"total\": "                  << total << ",\n";
	f << "    \"positive\": "               << pos   << ",\n";
	f << "    \"negative\": "               << neg   << ",\n";
	f << "    \"last_sensor_timestamp_us\": " << lastT << ",\n";
	f << "    \"note\": \"decoder-side counters, approximate; .raw is authoritative\"\n";
	f << "  }\n";
	f << "}\n";

	f.flush();
	if (!f) {
		errorOut = "write error while producing " + path;
		return false;
	}
	f.close();
	return true;
}

// ---------------------------------------------------------------------------
// SECTION 6 — main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
	Args args;
	std::string parseError;
	if (!ParseArgs(argc, argv, args, parseError)) {
		std::cerr << "[error] " << parseError << "\n\n";
		PrintUsage(argv[0]);
		return 2;
	}

	if (args.showHelp || argc == 1) {
		PrintUsage(argv[0]);
		return 0;
	}

	if (!args.infoOnly && args.outputPath.empty()) {
		std::cerr << "[error] --output is required (or use --info)\n\n";
		PrintUsage(argv[0]);
		return 2;
	}

	// Install handlers before opening the camera: a Ctrl-C during a slow
	// device open should not kill the process in a way that leaves the GigE
	// control channel claimed.
	std::signal(SIGINT,  HandleSignal);
	std::signal(SIGTERM, HandleSignal);

	// -----------------------------------------------------------------------
	// Open camera
	// -----------------------------------------------------------------------
	Metavision::Camera cam;
	try {
		if (args.serial.empty()) {
			std::cout << "[open] Opening first available camera...\n";
			cam = Metavision::Camera::from_first_available();
		} else {
			std::cout << "[open] Opening camera with serial " << args.serial << "...\n";
			cam = Metavision::Camera::from_serial(args.serial);
		}
	} catch (const Metavision::CameraException &e) {
		std::cerr << "[fatal] Could not open camera: " << e.what() << "\n"
		          << "        Checklist:\n"
		          << "          1. MV_HAL_PLUGIN_PATH includes LUCID's hal_plugin directory\n"
		          << "          2. LD_LIBRARY_PATH includes .../Metavision/lib\n"
		          << "          3. No Arena-based process (ArenaView, evs_recorder) holds\n"
		          << "             the camera -- only one process may claim the GigE Vision\n"
		          << "             control channel at a time\n";
		return 1;
	}

	// -----------------------------------------------------------------------
	// Bias profile — applied BEFORE start, since biases change what the
	// sensor generates, not how it is read out.
	// -----------------------------------------------------------------------
	if (!args.biasFileIn.empty()) {
		try {
			cam.biases().set_from_file(args.biasFileIn);
			std::cout << "[bias] Loaded bias profile: " << args.biasFileIn << "\n";
		} catch (const Metavision::CameraException &e) {
			// Hard failure on purpose. Silently recording with default biases
			// after being told to use a calibrated profile would produce data
			// that looks valid and is scientifically wrong.
			std::cerr << "[fatal] Failed to load bias file '" << args.biasFileIn
			          << "': " << e.what() << "\n";
			ExitNow(1);
		}
	}

	// -----------------------------------------------------------------------
	// ERC — after biases, before recording.
	//
	// --erc-off and --erc-rate are mutually exclusive in effect; --erc-off
	// wins if both are given, since disabling is the more conservative
	// request and is what calibration runs need (ERC drops events, which
	// masks the true noise floor).
	// -----------------------------------------------------------------------
	if (args.ercOff || args.ercRate > 0) {
		Metavision::I_ErcModule *erc = cam.get_device().get_facility<Metavision::I_ErcModule>();
		if (!erc) {
			std::cerr << "[warning] ERC was requested but this plugin exposes no I_ErcModule "
			             "facility. Continuing with the camera's current ERC state.\n";
		} else if (args.ercOff) {
			if (erc->enable(false)) {
				std::cout << "[erc] Disabled.\n";
			} else {
				std::cerr << "[warning] ERC disable was rejected by the device.\n";
			}
		} else {
			const uint32_t lo = erc->get_min_supported_cd_event_rate();
			const uint32_t hi = erc->get_max_supported_cd_event_rate();
			if (args.ercRate < lo || args.ercRate > hi) {
				std::cerr << "[fatal] --erc-rate " << args.ercRate
				          << " is outside the supported range [" << lo << ", " << hi
				          << "] events/sec for this camera.\n";
				ExitNow(2);
			}
			if (erc->set_cd_event_rate(args.ercRate) && erc->enable(true)) {
				std::cout << "[erc] Enabled at " << args.ercRate << " events/sec ("
				          << (args.ercRate / 1000000.0) << " Mev/s).\n";
			} else {
				std::cerr << "[warning] ERC configuration was rejected by the device; "
				             "continuing with its current state.\n";
			}
		}
	}

	// -----------------------------------------------------------------------
	// Introspect AFTER bias/ERC changes so what is reported and saved to the
	// sidecar is what is actually in effect, not what it was at open time.
	// -----------------------------------------------------------------------
	CameraInfo info = CollectCameraInfo(cam);
	PrintCameraInfo(info);

	if (!args.biasFileOut.empty()) {
		try {
			cam.biases().save_to_file(args.biasFileOut);
			std::cout << "[bias] Saved effective biases to: " << args.biasFileOut << "\n";
		} catch (const Metavision::CameraException &e) {
			std::cerr << "[warning] Could not save bias file '" << args.biasFileOut
			          << "': " << e.what() << "\n";
		}
	}

	if (args.infoOnly) {
		std::cout << "[info] --info requested; not recording.\n";
		ExitNow(0);
	}

	// -----------------------------------------------------------------------
	// Statistics callback.
	//
	// Runs on the SDK's decoding thread and receives a half-open [begin, end)
	// range per invocation, NOT one call per event. Kept to integer
	// arithmetic only: no I/O, no allocation, no locking. If this thread
	// stalls, the SDK's internal queues back up and events are lost.
	// -----------------------------------------------------------------------
	cam.cd().add_callback([](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
		uint64_t n = 0, pos = 0;
		int64_t  lastT = 0;
		for (const Metavision::EventCD *it = begin; it != end; ++it) {
			++n;
			if (it->p > 0) ++pos;
			lastT = static_cast<int64_t>(it->t);
		}
		if (n == 0) return;
		g_stats.totalEvents.fetch_add(n, std::memory_order_relaxed);
		g_stats.positiveEvents.fetch_add(pos, std::memory_order_relaxed);
		g_stats.negativeEvents.fetch_add(n - pos, std::memory_order_relaxed);
		g_stats.lastEventTimestampUs.store(lastT, std::memory_order_relaxed);
	});

	// -----------------------------------------------------------------------
	// Start recording BEFORE start(): the SDK writer must be attached when
	// the first events arrive, otherwise the head of the stream is lost.
	// -----------------------------------------------------------------------
	if (!cam.start_recording(args.outputPath)) {
		std::cerr << "[fatal] start_recording('" << args.outputPath << "') failed. "
		             "Check the path is writable and the directory exists.\n";
		ExitNow(1);
	}
	std::cout << "[rec] Writing sparse events to: " << args.outputPath << "\n";

	const std::string startedUtc = UtcTimestampString();
	const auto        startWall  = std::chrono::steady_clock::now();

	cam.start();

	if (args.durationSec > 0) {
		std::cout << "[rec] Recording for " << args.durationSec
		          << " s (Ctrl-C stops early).\n";
	} else {
		std::cout << "[rec] Recording until Ctrl-C.\n";
	}

	// -----------------------------------------------------------------------
	// Main loop. Polls rather than sleeps for the whole duration so Ctrl-C is
	// responsive within ~50 ms even on a multi-hour capture.
	// -----------------------------------------------------------------------
	std::string stopReason = "unknown";
	auto     lastStatsAt   = startWall;
	uint64_t lastStatsCount = 0;

	while (true) {
		if (g_stopRequested.load(std::memory_order_relaxed)) {
			stopReason = "signal (Ctrl-C / SIGTERM)";
			break;
		}
		if (!cam.is_running()) {
			// The SDK stopped on its own: device unplugged, link dropped, or
			// an internal error. Not a normal path, so it is named as such in
			// the sidecar rather than reported as a clean finish.
			stopReason = "camera stopped unexpectedly";
			break;
		}

		const auto now     = std::chrono::steady_clock::now();
		const double elapsed = std::chrono::duration<double>(now - startWall).count();

		if (args.durationSec > 0 && elapsed >= args.durationSec) {
			stopReason = "duration reached";
			break;
		}

		if (args.statsInterval > 0) {
			const double sinceStats = std::chrono::duration<double>(now - lastStatsAt).count();
			if (sinceStats >= args.statsInterval) {
				const uint64_t total = g_stats.totalEvents.load(std::memory_order_relaxed);
				const uint64_t delta = total - lastStatsCount;
				const double   rate  = (sinceStats > 0) ? (delta / sinceStats) : 0.0;

				std::cout << "[stats] t=" << std::fixed << std::setprecision(1) << elapsed << "s"
				          << "  events=" << total
				          << "  rate=" << std::setprecision(2) << (rate / 1e6) << " Mev/s"
				          << "  ON=" << g_stats.positiveEvents.load(std::memory_order_relaxed)
				          << "  OFF=" << g_stats.negativeEvents.load(std::memory_order_relaxed)
				          << std::endl;

				lastStatsAt    = now;
				lastStatsCount = total;
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}

	// -----------------------------------------------------------------------
	// Shutdown. Order matters: stop the camera first so no new events are
	// produced, then close the file, then report.
	// -----------------------------------------------------------------------
	std::cout << "\n[rec] Stopping (" << stopReason << ")...\n";

	try {
		cam.stop();
	} catch (const Metavision::CameraException &e) {
		std::cerr << "[warning] camera stop reported: " << e.what() << "\n";
	}

	try {
		if (!cam.stop_recording(args.outputPath)) {
			std::cerr << "[warning] stop_recording() returned false; the .raw may be "
			             "truncated. Verify it before relying on this run.\n";
		}
	} catch (const Metavision::CameraException &e) {
		std::cerr << "[warning] stop_recording reported: " << e.what() << "\n";
	}

	const std::string stoppedUtc  = UtcTimestampString();
	const double      wallSeconds =
		std::chrono::duration<double>(std::chrono::steady_clock::now() - startWall).count();

	// Refresh bias/ERC state for the sidecar: ERC in particular can be
	// changed by the device itself under load.
	info = CollectCameraInfo(cam);

	std::string sidecarError;
	if (WriteSidecar(args.outputPath, info, args, startedUtc, stoppedUtc,
	                 wallSeconds, stopReason, sidecarError)) {
		std::cout << "[meta] Wrote " << args.outputPath << ".meta.json\n";
	} else {
		std::cerr << "[warning] Sidecar not written: " << sidecarError << "\n";
	}

	const uint64_t total = g_stats.totalEvents.load(std::memory_order_relaxed);
	const uint64_t pos   = g_stats.positiveEvents.load(std::memory_order_relaxed);
	const uint64_t neg   = g_stats.negativeEvents.load(std::memory_order_relaxed);

	std::cout << "\n[done] " << args.outputPath << "\n"
	          << "[done] wall time        = " << std::fixed << std::setprecision(2)
	          << wallSeconds << " s\n"
	          << "[done] events observed  = " << total
	          << "  (ON=" << pos << ", OFF=" << neg << ")\n";
	if (wallSeconds > 0) {
		std::cout << "[done] average rate     = " << std::setprecision(3)
		          << (total / wallSeconds / 1e6) << " Mev/s\n";
	}

	if (total == 0) {
		std::cout << "[done] WARNING: zero events. Either nothing moved in the scene, or\n"
		             "       the lens cap is on, or biases are set far too insensitive.\n";
	}

	// See the file header: leaving through _Exit avoids a known
	// boost::lock_error abort inside the bundled libraries' static
	// destructors. All output is already durable at this point.
	ExitNow(0);
}
