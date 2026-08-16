// ============================================================================
//  evs_recorder.cpp
//  Standalone long-duration recorder for LUCID Triton2 EVS (TRT009S-EC)
//  Replaces ArenaView's recording function, which crashes on this camera.
//
//  Built using ONLY Arena SDK C++ APIs (namespace Arena). No Metavision SDK.
//
//  Every Arena/GenApi call used below is backed by a specific header citation
//  in the accompanying report (Phase 1/2). Where the exact behavior could NOT
//  be confirmed from the provided Include.zip (GenApi/GenICam headers were
//  not part of that zip; standard GigE Vision packet-size/resend node names
//  are not hardcoded anywhere in Arena's own headers), this is called out
//  explicitly in comments rather than silently assumed.
//
//  WHAT THIS RECORDER WRITES:
//    A custom binary container ("CAROEVT2"), NOT the Prophesee EVT3.0 wire
//    format. This recorder does NOT interpret payload bytes at capture time -
//    it copies them verbatim, as fast as possible, alongside a small per-buffer
//    header (frame id, device timestamp, host timestamp, timestamp source,
//    size). Interpretation into (x,y,t,p) arrays happens OFFLINE in
//    cevt_to_events.py, where mistakes are cheap to fix without re-running a
//    live camera. This keeps the hot path "dumb and fast", which is what a
//    never-block, never-crash recorder needs.
//
//  ESTABLISHED FACTS ABOUT THIS CAMERA (do not re-litigate without new data):
//    * There is NO sparse/async event output path. AcquisitionAccumulationMode
//      (TimeBased/EventBased) exists on the node map but reports
//      IsAvailable=false / IsReadable=false / IsWritable=false — locked in
//      firmware. --node-info and --set-enum both confirm this directly.
//      Over 2160 nodes were swept for Output/Frame/Mode/Event/XY/Decode/Pixel/
//      Accumulation/License/Feature/Format/Image: no unlock path exists via
//      GenICam introspection. This is a hardware/firmware limit, not a code bug.
//    * There is NO PixelFormat node, and no LucidXYTP128f. The XYPT chase is
//      over: TryApplyOutputFormat(), --output-format-node/value, --legacy-cdframe,
//      --strict-xypt and --flat-xypt have all been removed.
//    * EventFormat=EVT3_0 + EventFormatSize=Bpe16 is the one working config.
//      EventFormatSize is READ-ONLY and follows EventFormat on this firmware.
//    * Every payload is a DENSE ACCUMULATED FRAME of exactly width*height bytes
//      (1280*720 = 921600), NOT an EVT3.0 word stream. "EventFormat=EVT3_0"
//      names the sensor<->FPGA protocol, not the GigE payload format.
//
//  CONSEQUENCE FOR TIME — READ THIS BEFORE TRUSTING ANY TIMESTAMP:
//    A dense accumulated frame asserts only "this pixel fired somewhere inside
//    this accumulation window". Sub-window event ordering is destroyed inside
//    the camera. No per-event microsecond timestamp exists, here or offline,
//    and any code that produces one is inventing it.
//    What this recorder therefore guarantees is narrower but honest: every
//    record carries a MEASURED window time (device clock if the camera provides
//    one, monotonic host arrival time otherwise) plus an explicit
//    timestampSource tag, and the file header carries the camera's own
//    AcquisitionFrameRate/AcquisitionFrameTime so the offline converter never
//    has to be told the frame rate by hand. See the SECTION 1 comment block.
// ============================================================================

// NOMINMAX: Windows only — windows.h (pulled in by Arena SDK on Windows) would
// #define min/max as macros, swallowing std::max calls. Not needed on Linux/macOS.
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#endif

#include <Arena/ArenaApi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// SECTION 0 — Global shutdown flag + signal handling
// ============================================================================
// A single atomic<bool> is the only thing the signal handler touches. Both
// worker threads poll it. This is the standard, safe pattern for graceful
// shutdown in C++: signal handlers must not do anything non-trivial (no
// I/O, no locks, no allocation), so all it does is flip a flag.
// ----------------------------------------------------------------------------

std::atomic<bool> g_stopRequested{ false };

// Set by the writer thread if any disk write fails. Read by main() so the
// process exits non-zero — a shell script driving a batch of recordings must be
// able to tell a truncated file from a good one without parsing stdout.
std::atomic<bool> g_writeFailed{ false };

// Countdown for the Step-0 per-buffer diagnostic (see AcquisitionThreadFunc).
// Not atomic: only ever touched from the single acquisition thread.
// Set from --debug-buffers [N] (default N=20 when the flag is passed bare).
int g_debugBufferCount = 0;

// Monotonic origin for every hostRecvNs written into a RecordHeader. Latched
// once in main() immediately before StartStream(), then only ever read. Using
// steady_clock (not system_clock) is deliberate: NTP slew or a manual clock
// change mid-recording must not make event time appear to run backwards. The
// wall-clock equivalent of this instant is stored once in
// FileHeader::hostUnixEpochNsAtStart for cross-device correlation.
std::chrono::steady_clock::time_point g_hostT0;

uint64_t HostNsSinceStart()
{
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::steady_clock::now() - g_hostT0).count());
}

void SignalHandler(int /*signum*/)
{
	g_stopRequested.store(true, std::memory_order_relaxed);
}

// ============================================================================
// SECTION 1 — On-disk container format (CAROECT-D custom format)
// ============================================================================
// IMPORTANT: this is NOT the Prophesee EVT3.0 raw wire format. It is a
// minimal, self-describing container designed so that nothing about the
// capture is lost or guessed at write time. All interpretation of the event
// payload bytes (casting to LucidXYTPPixel / EvsRawDecodedEvent) is deferred
// to the offline Python converter described in Phase 5 of the report.
//
// File layout:
//   [FileHeader]                  (written once, at the start)
//   [RecordHeader][payload bytes] (repeated once per Arena buffer)
//   [RecordHeader][payload bytes]
//   ...
// ----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// FORMAT VERSION 2 ("CAROEVT2") — WHY IT EXISTS
// ---------------------------------------------------------------------------
// V1 ("CAROEVT1") stored exactly one time field per record: `timestampNs`,
// populated ONLY when Arena reported HasImageData()==true. On this camera that
// branch was observed never to fire, so every V1 record carried timestampNs=0,
// and the offline converter had no choice but to invent time as
// `frame_index / --fps` with an fps the operator typed in by hand. That is the
// "fabricated timestamp" problem: a guessed number, indistinguishable in the
// output file from a measured one, silently flowing into simulator calibration.
//
// V2 fixes the *information* problem, not by inventing a better guess, but by
// recording every real time source the hardware actually offers and labelling
// which one was used, per record:
//
//   1. deviceTimestampNs — Arena's buffer-level device clock, IF obtainable.
//      V1 only tried this when HasImageData() was true. V2 attempts the
//      AsImage() cast unconditionally (guarded), because Arena's own docs say
//      GetPayloadType()==BufferPayloadTypeImage(0x1) implies HasImageData()==true,
//      which directly contradicts the field observation of
//      "payloadType=1 HasImageData=0". One of the two is wrong; rather than pick
//      a side, V2 probes both and writes down what actually happened.
//
//   2. hostRecvNs — monotonic host time (steady_clock) at the moment the buffer
//      was handed to us, relative to a t0 latched at StartStream. This is ALWAYS
//      available. It is arrival time, not sensor time — it carries network and
//      scheduling jitter — but it is *measured*, and it is monotonic, so
//      inter-frame spacing from it is real data rather than an assumption.
//
//   3. acquisitionFrameRateHz / acquisitionFrameTimeUs (FileHeader) — read
//      straight off the camera's own AcquisitionControl nodes before streaming.
//      This is the accumulation-window length that the `--fps 30` guess was
//      standing in for. Recording it removes the guess entirely.
//
//   4. deviceTimestampAtStartNs (FileHeader) — the device's own Timestamp
//      counter, latched via TimestampLatch/TimestampLatchValue right after
//      TimestampReset, so host time and device time share a known origin.
//
// Nothing here manufactures per-event microsecond time — that information does
// not exist in a dense accumulated frame and no amount of code can create it.
// What V2 guarantees is that the offline side can always tell the difference
// between a measured timestamp and a synthesised one.
// ---------------------------------------------------------------------------

// Values for RecordHeader::timestampSource. Written per record so the offline
// converter never has to guess where a number came from.
enum : uint8_t
{
	kTimestampSourceNone   = 0, // no time available at all (should not happen in V2)
	kTimestampSourceDevice = 1, // deviceTimestampNs is a real device-clock reading
	kTimestampSourceHost   = 2  // only hostRecvNs is real; deviceTimestampNs is 0
};

// FileHeader::flags bits.
enum : uint32_t
{
	kFileFlagTimestampResetOk = 1u << 0, // TimestampReset command executed successfully
	kFileFlagDeviceTsAvailable = 1u << 1, // at least the first buffer yielded a device timestamp
	kFileFlagFrameRateKnown   = 1u << 2  // acquisitionFrameRateHz was read from the camera
};

#pragma pack(push, 1)

struct FileHeader
{
	char     magic[8];              // "CAROEVT2" (V1 files carry "CAROEVT1" and a 36-byte header)
	uint64_t pixelFormat;           // Arena::IImage::GetPixelFormat(), or GetPayloadType() if not image-typed
	uint32_t bitsPerPixel;          // Arena::IImage::GetBitsPerPixel(), 0 if unknown
	uint32_t width;                 // Arena::IImage::GetWidth(), 0 if unknown
	uint32_t height;                // Arena::IImage::GetHeight(), 0 if unknown
	uint32_t recordHeaderSize;      // sizeof(RecordHeader) — self-describing, so a reader
	                                // can skip records it does not understand
	double   acquisitionFrameRateHz;    // camera's AcquisitionFrameRate node, 0 if unreadable
	uint64_t acquisitionFrameTimeUs;    // camera's AcquisitionFrameTime node, 0 if unreadable
	uint64_t hostUnixEpochNsAtStart;    // system_clock at t0 — wall-clock anchor for cross-device sync
	uint64_t deviceTimestampAtStartNs;  // device Timestamp latched at t0, 0 if unreadable
	uint32_t flags;                     // kFileFlag* bitmask
	uint32_t reserved;                  // zero-filled
};

struct RecordHeader
{
	uint64_t frameId;             // Arena::IBuffer::GetFrameId() — gap detection, NOT time
	uint64_t deviceTimestampNs;   // device clock (buffer-level, NOT per-event); 0 if unavailable
	uint64_t hostRecvNs;          // monotonic host time since recorder t0; always valid
	uint64_t payloadSize;         // Arena::IBuffer::GetSizeFilled(); payload bytes follow
	uint8_t  timestampSource;     // kTimestampSource*
	uint8_t  reserved[7];         // zero-filled, keeps the struct 8-byte aligned
};

#pragma pack(pop)

// 8 + 8 + 4 + 4 + 4 + 4 + 8 + 8 + 8 + 8 + 4 + 4 = 72 bytes, packed.
static_assert(sizeof(FileHeader) == 72, "FileHeader size changed - update the Python readers too");
static_assert(sizeof(RecordHeader) == 40, "RecordHeader size changed - update the Python readers too");

// ============================================================================
// SECTION 2 — Lock-free single-producer/single-consumer ring buffer
// ============================================================================
// Thread A (acquisition) is the ONLY producer. Thread B (disk writer) is the
// ONLY consumer. This is the classic SPSC ring buffer: each side only ever
// writes its own atomic index and only ever reads the other side's atomic
// index, so no compare-and-swap and no mutex are needed on the data path.
//
// Slots are pre-allocated once at startup (sized from the stream's
// 'PayloadSize' node, see main()) so the hot path never allocates memory.
// If the queue is full, TryPush() returns false immediately (never blocks);
// the caller is responsible for counting this as a drop and moving on.
// ----------------------------------------------------------------------------

class SpscByteRing
{
public:
	SpscByteRing(size_t numSlots, size_t slotCapacity)
		: m_numSlots(numSlots)
		, m_slotCapacity(slotCapacity)
		, m_slots(numSlots)
		, m_validSize(numSlots, 0)
		, m_recordHeader(numSlots)
		, m_head(0)
		, m_tail(0)
	{
		if (numSlots < 2)
			throw std::invalid_argument("SpscByteRing needs at least 2 slots");
		for (auto& slot : m_slots)
			slot.resize(slotCapacity);
	}

	size_t Capacity() const { return m_numSlots; }
	size_t SlotBytes() const { return m_slotCapacity; }

	// Producer-only. Never blocks. Returns false if the ring is full.
	bool TryPush(const uint8_t* data, size_t size, const RecordHeader& hdr)
	{
		const size_t head = m_head.load(std::memory_order_relaxed);
		const size_t nextHead = (head + 1) % m_numSlots;
		const size_t tail = m_tail.load(std::memory_order_acquire);

		if (nextHead == tail)
			return false; // full

		size_t copySize = size;
		bool truncated = false;
		if (copySize > m_slotCapacity)
		{
			copySize = m_slotCapacity;
			truncated = true;
		}

		std::memcpy(m_slots[head].data(), data, copySize);
		m_validSize[head] = copySize;
		m_recordHeader[head] = hdr;
		m_recordHeader[head].payloadSize = copySize; // reflect actual bytes stored

		m_head.store(nextHead, std::memory_order_release);

		if (truncated)
			m_truncationCount.fetch_add(1, std::memory_order_relaxed);

		return true;
	}

	// Consumer-only. Returns false if the ring is empty.
	bool TryPop(std::vector<uint8_t>& outBuf, size_t& outSize, RecordHeader& outHdr)
	{
		const size_t tail = m_tail.load(std::memory_order_relaxed);
		const size_t head = m_head.load(std::memory_order_acquire);

		if (tail == head)
			return false; // empty

		outSize = m_validSize[tail];
		outHdr = m_recordHeader[tail];
		if (outBuf.size() < outSize)
			outBuf.resize(outSize);
		std::memcpy(outBuf.data(), m_slots[tail].data(), outSize);

		const size_t nextTail = (tail + 1) % m_numSlots;
		m_tail.store(nextTail, std::memory_order_release);
		return true;
	}

	bool Empty() const
	{
		return m_tail.load(std::memory_order_acquire) == m_head.load(std::memory_order_acquire);
	}

	uint64_t TruncationCount() const { return m_truncationCount.load(std::memory_order_relaxed); }

private:
	size_t m_numSlots;
	size_t m_slotCapacity;
	std::vector<std::vector<uint8_t>> m_slots;
	std::vector<size_t> m_validSize;
	std::vector<RecordHeader> m_recordHeader;
	std::atomic<size_t> m_head;
	std::atomic<size_t> m_tail;
	std::atomic<uint64_t> m_truncationCount{ 0 };
};

// ============================================================================
// SECTION 3 — Live statistics counters
// ============================================================================

struct Stats
{
	std::atomic<uint64_t> buffersAcquired{ 0 };
	std::atomic<uint64_t> buffersRequeued{ 0 };
	std::atomic<uint64_t> incompleteBuffers{ 0 };
	std::atomic<uint64_t> queueOverflowDrops{ 0 };
	std::atomic<uint64_t> bytesWritten{ 0 };
	std::atomic<uint64_t> recordsWritten{ 0 };
	std::atomic<uint64_t> reconnectEvents{ 0 };
};

Stats g_stats;

// ============================================================================
// SECTION 4 — Disconnect callback (Arena::IDisconnectCallback)
// ============================================================================
// Confirmed API: Arena/IDisconnectCallback.h declares
//   virtual void OnDeviceDisconnected(Arena::IDevice* pDisconnectedDevice) = 0;
// Registered via Arena::ISystem::RegisterDeviceDisconnectCallback (ISystem.h).
// This does not itself recover the connection - it only logs. Recovery is
// handled in the acquisition thread's exception handling (Section 6), using
// the confirmed Arena::IDevice::WaitForReconnection(uint64_t timeout) API.
// ----------------------------------------------------------------------------

class RecorderDisconnectCallback : public Arena::IDisconnectCallback
{
public:
	void OnDeviceDisconnected(Arena::IDevice* /*pDisconnectedDevice*/) override
	{
		std::cerr << "[warning] Device disconnected unexpectedly (network/PoE hiccup?). "
			<< "Acquisition thread will attempt to wait for reconnection." << std::endl;
	}
};

// ============================================================================
// SECTION 5 — GenApi node helpers
// ============================================================================
// NOTE ON PROVENANCE: the GenApi::CEnumerationPtr / CEnumEntryPtr / CIntegerPtr
// usage pattern below is copied and adapted directly from the example code
// embedded in Arena/IDevice.h's own Doxygen comments (see Phase 2 of the
// report for the exact quoted block). The GenApi/GenICam headers themselves
// were NOT part of the provided Include.zip, so the precise class names
// could not be independently verified against that zip - if your build
// reports these types as undeclared, add:
//     #include <GenApi/GenApi.h>
// from your ArenaSDK's GenICam/library/... directory.
// ----------------------------------------------------------------------------

// EMPIRICAL FINDING (from --dump-nodes against the real TRT009S-E): this
// camera does NOT expose a standard SFNC "PixelFormat" node at all. The real
// feature controlling event output is "EventFormat", a generic enumeration
// with exactly two entries matching the actual Prophesee protocol names:
//   EVT2_1   (EnumEntry_EventFormat_EVT2_1)
//   EVT3_0   (EnumEntry_EventFormat_EVT3_0)
// A second, related node "EventFormatSize" selects a bit-width variant:
//   Bpe16    (EnumEntry_EventFormatSize_Bpe16)
//   Bpe64    (EnumEntry_EventFormatSize_Bpe64)
// ("Bpe" = bits/bytes per event - exact meaning not confirmed from headers;
// discovered empirically, not guessed). Everything below targets these two
// real nodes instead of the nonexistent "PixelFormat".
void ListEventFormats(GenApi::INodeMap* pNodeMap)
{
	auto printEnum = [&](const char* nodeName)
	{
		GenApi::CEnumerationPtr pEnum = pNodeMap->GetNode(nodeName);
		if (!pEnum.IsValid())
		{
			std::cout << nodeName << ": node not found on this device." << std::endl;
			return;
		}
		GenApi::StringList_t symbolics;
		pEnum->GetSymbolics(symbolics);
		std::cout << nodeName << " entries:\n";
		for (auto& s : symbolics)
			std::cout << "  " << s.c_str() << "\n";
	};

	printEnum("EventFormat");
	printEnum("EventFormatSize");

	std::cout << "\nPass a value with --event-format <NAME> (and optionally\n"
		<< "--event-format-size <NAME>) to record. EVT3_0 is the modern,\n"
		<< "denser Prophesee protocol and is the recommended starting point.\n"
		<< std::endl;
}

// Diagnostic: dump every node name on the device (unchanged from before -
// kept because it is how EventFormat/EventFormatSize were found in the
// first place, and remains useful if yet another node needs discovering).
void DumpAllNodes(GenApi::INodeMap* pNodeMap, const std::string& filterSubstring)
{
	GenApi::NodeList_t nodes;
	pNodeMap->GetNodes(nodes);

	std::cout << "Total nodes on device node map: " << nodes.size() << "\n";
	if (!filterSubstring.empty())
		std::cout << "Filtering to names containing (case-sensitive): \"" << filterSubstring << "\"\n";
	std::cout << "----\n";

	size_t printedCount = 0;
	for (auto* pNode : nodes)
	{
		if (pNode == nullptr)
			continue;

		const std::string name = pNode->GetName().c_str();
		if (!filterSubstring.empty() && name.find(filterSubstring) == std::string::npos)
			continue;

		std::cout << name << "  [interfaceType=" << pNode->GetPrincipalInterfaceType() << "]\n";
		++printedCount;
	}

	std::cout << "----\n" << printedCount << " node(s) printed." << std::endl;
}

// Sets an enumeration node by symbolic name if the node exists; if optional
// and absent, does nothing (silently) rather than throwing.
//
// IMPORTANT — EventFormatSize is NOT independently writable on this camera:
// a real run hit "[fatal] GenICam exception: Node is not writable." on the
// SetIntValue() call for EventFormatSize, immediately after EventFormat was
// set and confirmed successfully. This matches an earlier observation: after
// setting ONLY EventFormat=EVT3_0 (EventFormatSize never touched), --get-enum
// EventFormatSize already reported Bpe16. Put together, the simplest
// explanation is that EventFormatSize is a READ-ONLY reflection driven by
// whichever EventFormat entry is selected (EVT3_0 -> Bpe16 on this firmware),
// not an independently settable node — the earlier "stale leftover value"
// theory (see Stage 1.1 history below) was the wrong explanation for the
// same symptom. So: try to WRITE only if the node reports writable right
// now; if it doesn't, fall back to READ-and-verify — if the camera already
// reports the requested value we're done (that's the expected, healthy case
// for EventFormatSize), and only throw if it reports something ELSE (a
// genuine mismatch we can't fix by writing).
void ApplyEnumIfPresent(GenApi::INodeMap* pNodeMap, const char* nodeName,
                        const std::string& value, bool required)
{
	GenApi::CEnumerationPtr pEnum = pNodeMap->GetNode(nodeName);
	if (!pEnum.IsValid())
	{
		if (required)
			throw std::runtime_error(std::string(nodeName) + " node not found on device node map.");
		return;
	}

	GenApi::CEnumEntryPtr pEntry = pEnum->GetEntryByName(value.c_str());
	if (!pEntry.IsValid())
	{
		throw std::runtime_error(
			std::string(nodeName) + " entry '" + value + "' not found. Run --list-event-formats "
			"first to see the exact symbolic names this camera's firmware exposes.");
	}

	if (!GenApi::IsWritable(pEnum))
	{
		// Not writable right now — could be permanently read-only (driven by
		// another node, e.g. EventFormatSize following EventFormat), or
		// temporarily locked by device state. Either way, don't guess:
		// verify what the camera actually reports and decide from that.
		int64_t currentVal = pEnum->GetIntValue();
		if (currentVal == pEntry->GetValue())
		{
			std::cout << nodeName << " = " << value << " (already set — node is read-only "
				<< "on this camera right now, but the current value already matches what "
				<< "was requested, so this is fine)" << std::endl;
			return;
		}
		throw std::runtime_error(
			std::string(nodeName) + " is not writable right now, and its current value does "
			"NOT match the requested '" + value + "' (raw int " +
			std::to_string(currentVal) + " vs expected " + std::to_string(pEntry->GetValue()) +
			"). If this is EventFormatSize, it may be entirely driven by the current "
			"EventFormat selection rather than independently settable on this firmware — "
			"try a different --event-format-size value that matches what this EventFormat "
			"actually produces (check via --get-enum " + std::string(nodeName) +
			" right after setting --event-format alone), or drop --event-format-size "
			"and rely on the read-back verification instead of a hard requirement.");
	}

	pEnum->SetIntValue(pEntry->GetValue());

	// Confirm the camera actually accepted the value instead of trusting
	// SetIntValue() blindly — a stale/leftover value from a previous session
	// (see the Bpe16-leftover incident: EventFormatSize silently kept its
	// old value from a prior run when this node wasn't touched) can look
	// identical to a successful set unless we read it back.
	int64_t confirmVal = pEnum->GetIntValue();
	if (confirmVal != pEntry->GetValue())
	{
		throw std::runtime_error(
			std::string(nodeName) + ": set to '" + value + "' failed — camera still reports "
			"raw int value " + std::to_string(confirmVal) + " (expected " +
			std::to_string(pEntry->GetValue()) + "). Refusing to continue with an ambiguous "
			"node state.");
	}
	std::cout << "Set " << nodeName << " = " << value << " (confirmed)" << std::endl;
}

// Prints the CURRENT symbolic value of an enum node — deliberately does NOT
// assume an unverified "GetCurrentEntry()" method exists. Instead it uses
// only already-confirmed APIs: read the current raw int (GetIntValue,
// confirmed in GenApi/IEnumeration.h), then walk every symbolic name
// (GetSymbolics, confirmed) and match by looking up each one's int value
// (GetEntryByName + IEnumEntry::GetValue, both confirmed) until one matches.
void PrintEnumCurrentValue(GenApi::INodeMap* pNodeMap, const char* nodeName)
{
	GenApi::CEnumerationPtr pEnum = pNodeMap->GetNode(nodeName);
	if (!pEnum.IsValid())
	{
		std::cout << nodeName << ": node not found on this device." << std::endl;
		return;
	}

	int64_t currentInt = pEnum->GetIntValue();
	GenApi::StringList_t symbolics;
	pEnum->GetSymbolics(symbolics);

	for (auto& s : symbolics)
	{
		GenApi::CEnumEntryPtr pEntry = pEnum->GetEntryByName(s);
		if (pEntry.IsValid() && pEntry->GetValue() == currentInt)
		{
			std::cout << nodeName << " = " << s.c_str() << "  (raw int value " << currentInt << ")"
				<< std::endl;
			return;
		}
	}
	std::cout << nodeName << " = <unknown symbolic>  (raw int value " << currentInt << ")" << std::endl;
}

// Prints what payload/interface type the first captured buffer actually is.
// We no longer assume BufferPayloadTypeImage + HasImageData() - this prints
// the ground truth so the rest of the pipeline can be adjusted if needed.
void DiagnoseFirstBuffer(Arena::IBuffer* pBuffer)
{
	std::cout << "First buffer diagnostic: "
		<< "payloadType=" << pBuffer->GetPayloadType()
		<< " hasImageData=" << pBuffer->HasImageData()
		<< " hasChunkData=" << pBuffer->HasChunkData()
		<< " sizeFilled=" << pBuffer->GetSizeFilled()
		<< " frameId=" << pBuffer->GetFrameId()
		<< std::endl;
}

// Explicitly select 'OldestFirst' (not the device default 'OldestFirstOverwrite')
// so that a full output queue produces an honest, countable drop
// ('StreamLostFrameCount') instead of silently overwriting unread data.
// Node name and the three enum values are confirmed verbatim from the

// Doxygen example embedded in Arena/IDevice.h (StartStream docblock).
void SetStreamBufferHandlingModeOldestFirst(GenApi::INodeMap* pTLStreamNodeMap)
{
	GenApi::CEnumerationPtr pMode = pTLStreamNodeMap->GetNode("StreamBufferHandlingMode");
	if (!pMode.IsValid())
	{
		std::cerr << "[warning] 'StreamBufferHandlingMode' node not found; "
			<< "proceeding with device default." << std::endl;
		return;
	}
	GenApi::CEnumEntryPtr pEntry = pMode->GetEntryByName("OldestFirst");
	if (!pEntry.IsValid())
	{
		std::cerr << "[warning] 'OldestFirst' entry not found on 'StreamBufferHandlingMode'; "
			<< "proceeding with device default." << std::endl;
		return;
	}
	pMode->SetIntValue(pEntry->GetValue());
}

// 'PayloadSize' node name confirmed verbatim from Arena/IBuffer.h's
// IsIncomplete() docblock. Used only to size our own ring buffer slots -
// never assumed to be the exact size of every buffer (event cameras produce
// variable-size payloads depending on how many events occurred).
uint64_t GetStreamPayloadSize(GenApi::INodeMap* pTLStreamNodeMap)
{
	try
	{
		GenApi::CIntegerPtr pNode = pTLStreamNodeMap->GetNode("PayloadSize");
		if (pNode.IsValid())
			return static_cast<uint64_t>(pNode->GetValue());
	}
	catch (...)
	{
		// Node absent or unreadable on this GenTL producer version; fall back below.
	}
	return 0;
}

// 'StreamMissedPacketCount' and 'StreamLostFrameCount' node names confirmed
// verbatim from Arena/IBuffer.h and Arena/IDevice.h docblocks respectively.
uint64_t ReadStreamIntegerNode(GenApi::INodeMap* pTLStreamNodeMap, const char* nodeName)
{
	try
	{
		GenApi::CIntegerPtr pNode = pTLStreamNodeMap->GetNode(nodeName);
		if (pNode.IsValid())
			return static_cast<uint64_t>(pNode->GetValue());
	}
	catch (...)
	{
		// Node absent on this GenTL producer version; treat as unavailable (0).
	}
	return 0;
}

// Sets a boolean node to `value` IF it exists and is writable; otherwise
// prints a one-line status and does nothing. Never throws, never crashes.
//
// PROVENANCE NOTE: unlike every other node name used elsewhere in this file
// (EventFormat, Width, Height, StreamBufferHandlingMode, PayloadSize,
// StreamMissedPacketCount, StreamLostFrameCount - all confirmed either by
// --dump-nodes against the real camera or by a quoted Arena SDK header
// docblock), "StreamPacketResendEnable" and "StreamAutoNegotiatePacketSize"
// have NOT been confirmed against this camera's TL Stream node map or any
// Arena header. They are common GenTL-standard names on many GigE Vision
// cameras, but that is not proof they exist HERE. Run this to check first:
//     ./evs_recorder --dump-nodes --target stream --filter Packet
// If the names differ, this function will simply print "not found" and skip
// - it will not crash or silently do the wrong thing either way.
bool TryEnableBoolean(GenApi::INodeMap* pNodeMap, const char* nodeName, bool value)
{
	try
	{
		GenApi::CBooleanPtr pNode = pNodeMap->GetNode(nodeName);
		if (!pNode.IsValid())
		{
			std::cerr << "[net-tuning] '" << nodeName << "' not found on this node map "
				<< "- skipped (not fatal). Run --dump-nodes --target stream --filter "
				<< "Packet to find the real name if one exists." << std::endl;
			return false;
		}
		if (!GenApi::IsWritable(pNode))
		{
			std::cerr << "[net-tuning] '" << nodeName << "' exists but is not writable "
				<< "right now - skipped." << std::endl;
			return false;
		}
		pNode->SetValue(value);
		std::cout << "[net-tuning] " << nodeName << " = " << (value ? "true" : "false")
			<< std::endl;
		return true;
	}
	catch (GenICam::GenericException& e)
	{
		std::cerr << "[net-tuning] '" << nodeName << "': " << e.GetDescription()
			<< " - skipped (not fatal)." << std::endl;
		return false;
	}
}

// XYPT was chased across two rounds of debugging and is now DEAD for this
// camera/firmware — confirmed empirically: the node this program tried
// ("EvsOutputFormat") does not exist on the TRT009S-E's node map (see
// --dump-nodes --filter Output / Frame / Mode / XY output, all negative),
// and there is no other candidate node exposing an XYPT-like enum on this
// firmware. TryApplyOutputFormat(), --output-format-node/value,
// --legacy-cdframe, --strict-xypt and --flat-xypt have all been removed —
// this program now ALWAYS uses the camera's one confirmed real decode path
// (EventFormat=EVT3_0, EventFormatSize=Bpe16/Bpe64), and per-event real
// microsecond timestamps are recovered entirely offline in
// cevt_to_events.py by decoding the EVT3.0 TIME_LOW/TIME_HIGH words inside
// the payload — never from a camera-side XYPT struct that does not exist.

// --erc-rate-limit <Mev/s>: ErcRateLimit is a FLOAT node (interfaceType=5,
// confirmed via --dump-nodes against the real TRT009S-E on 2026-07-xx) -
// NOT an Integer like ErcReferencePeriod (interfaceType=2). Using the wrong
// GenApi pointer type (CIntegerPtr) here would silently fail IsValid() and
// do nothing - CFloatPtr is required. Confirmed real API: GenApi::CFloatPtr,
// IFloat::SetValue(double, bool=true) / GetValue(bool=false, bool=false)
// (GenApi/Pointer.h, GenApi/IFloat.h).
bool TrySetFloat(GenApi::INodeMap* pNodeMap, const char* nodeName, double value)
{
	try
	{
		GenApi::CFloatPtr pNode = pNodeMap->GetNode(nodeName);
		if (!pNode.IsValid())
		{
			std::cerr << "[erc] '" << nodeName << "' not found or not a Float node - skipped."
				<< std::endl;
			return false;
		}
		if (!GenApi::IsWritable(pNode))
		{
			std::cerr << "[erc] '" << nodeName << "' exists but is not writable right now - "
				<< "skipped." << std::endl;
			return false;
		}
		pNode->SetValue(value);
		std::cout << "[erc] " << nodeName << " = " << value << std::endl;
		return true;
	}
	catch (GenICam::GenericException& e)
	{
		std::cerr << "[erc] '" << nodeName << "': " << e.GetDescription()
			<< " - skipped (not fatal)." << std::endl;
		return false;
	}
}

double ReadFloatNode(GenApi::INodeMap* pNodeMap, const char* nodeName)
{
	try
	{
		GenApi::CFloatPtr pNode = pNodeMap->GetNode(nodeName);
		// IsReadable() matters here: a node can exist and be the right type while
		// still being gated off by device state. GetValue() on such a node throws,
		// and the catch below would turn that into the same -1.0 sentinel as
		// "absent" — losing the distinction. Checking first keeps the fast path
		// exception-free and makes the sentinel mean exactly one thing.
		if (pNode.IsValid() && GenApi::IsReadable(pNode))
			return pNode->GetValue();
	}
	catch (...) {}
	return -1.0; // sentinel: node absent/unreadable
}

// Reads an integer node from the DEVICE node map. Distinct from
// ReadStreamIntegerNode() only in intent (that one is for TL-stream counters
// where 0 is a sane "unavailable"), but here 0 is a legitimate value — e.g. the
// device Timestamp right after TimestampReset — so absence is reported through
// the return flag instead of being folded into the value.
bool TryReadIntegerNode(GenApi::INodeMap* pNodeMap, const char* nodeName, uint64_t& outValue)
{
	try
	{
		GenApi::CIntegerPtr pNode = pNodeMap->GetNode(nodeName);
		if (!pNode.IsValid() || !GenApi::IsReadable(pNode))
			return false;
		outValue = static_cast<uint64_t>(pNode->GetValue());
		return true;
	}
	catch (GenICam::GenericException&)
	{
		return false;
	}
}

// Executes a GenICam command node (interfaceType=4) if present and writable.
// GenApi::CCommandPtr + ->Execute() is confirmed real API: it appears verbatim
// in Arena's own Doxygen example inside ArenaApi headers
// ("GenApi::CCommandPtr pTestEventGenerate = pNodeMap->GetNode(...); ...
// pTestEventGenerate->Execute();"). Never fatal — a camera that refuses
// TimestampReset is still perfectly recordable, we just lose the shared origin.
bool TryExecuteCommand(GenApi::INodeMap* pNodeMap, const char* nodeName)
{
	try
	{
		GenApi::CCommandPtr pCmd = pNodeMap->GetNode(nodeName);
		if (!pCmd.IsValid())
		{
			std::cerr << "[clock] '" << nodeName << "' not found - skipped (not fatal)." << std::endl;
			return false;
		}
		if (!GenApi::IsWritable(pCmd))
		{
			std::cerr << "[clock] '" << nodeName << "' exists but is not writable right now - skipped."
				<< std::endl;
			return false;
		}
		pCmd->Execute();
		std::cout << "[clock] executed " << nodeName << std::endl;
		return true;
	}
	catch (GenICam::GenericException& e)
	{
		std::cerr << "[clock] '" << nodeName << "': " << e.GetDescription()
			<< " - skipped (not fatal)." << std::endl;
		return false;
	}
}

// Latches and reads the device's own free-running Timestamp counter.
// Confirmed present on this camera's node map (--dump-nodes: Timestamp,
// TimestampReset, TimestampLatch, TimestampLatchValue at interfaceType 2/4/4/2).
// TimestampLatch is a command that snapshots the counter into
// TimestampLatchValue; some firmwares also allow reading `Timestamp` directly,
// so both are tried.
bool TryLatchDeviceTimestampNs(GenApi::INodeMap* pNodeMap, uint64_t& outNs)
{
	if (TryExecuteCommand(pNodeMap, "TimestampLatch"))
	{
		if (TryReadIntegerNode(pNodeMap, "TimestampLatchValue", outNs))
			return true;
	}
	return TryReadIntegerNode(pNodeMap, "Timestamp", outNs);
}

// ---------------------------------------------------------------------------
// Buffer-level device timestamp, probed rather than assumed.
// ---------------------------------------------------------------------------
// The V1 code did:
//     if (pBuffer->HasImageData()) { hdr.timestampNs = pBuffer->AsImage()->GetTimestampNs(); }
// and that branch reportedly never fired on this camera, despite
// GetPayloadType() returning 1. Arena's own header documents
// BufferPayloadTypeImage = 0x0001 and states HasImageData() returns true
// exactly when the payload type is BufferPayloadTypeImage or
// BufferPayloadTypeImageExtendedChunk. So "payloadType=1 && !HasImageData()"
// is self-contradictory under the documented contract — meaning either the
// docs, the GenTL producer, or the earlier observation is wrong.
//
// Rather than bet on which, this helper ignores HasImageData() as a gate and
// treats it purely as a hint: it attempts the AsImage() cast directly, honours
// the documented "Null on failure" contract, and swallows any GenICam
// exception. Worst case it costs one failed cast per buffer; best case it
// recovers a real device clock that V1 was throwing away.
bool TryGetBufferTimestampNs(Arena::IBuffer* pBuffer, uint64_t& outNs)
{
	if (pBuffer == nullptr)
		return false;
	try
	{
		Arena::IImage* pImage = pBuffer->AsImage();
		if (pImage == nullptr)
			return false;
		const uint64_t ts = pImage->GetTimestampNs();
		// A device that does not populate the payload leader timestamp reports 0
		// forever. Treating 0 as "got a real timestamp" is precisely how V1's
		// zeros ended up masquerading as data, so reject it here.
		if (ts == 0)
			return false;
		outNs = ts;
		return true;
	}
	catch (GenICam::GenericException&)
	{
		return false;
	}
	catch (std::exception&)
	{
		return false;
	}
}

// --load-features <file.txt>: applies a previously-saved bias/exposure
// profile to the device's main node map BEFORE StartStream, using Arena's
// own Arena::FeatureStream::Read() (confirmed real API, Arena/FeatureStream.h).
// This is what makes "one command = one fully-specified recording session"
// possible instead of silently inheriting whatever was last written to the
// camera (e.g. via ArenaView) from an unknown prior session.
void LoadFeaturesIfRequested(GenApi::INodeMap* pNodeMap, const std::string& path)
{
	if (path.empty())
		return;
	std::cout << "Loading feature profile: " << path << std::endl;
	Arena::FeatureStream fs(pNodeMap);
	fs.Read(path.c_str());   // throws GenICam::GenericException on failure; let caller's
	                 // top-level catch handle it - do not mask a bad profile load.
	std::cout << "Feature profile loaded OK." << std::endl;
}

// ============================================================================
// SECTION 6 — Thread A: acquisition
// ============================================================================

void AcquisitionThreadFunc(Arena::IDevice* pDevice, SpscByteRing& ring, std::atomic<bool>& stopFlag)
{
	// Short timeout so the loop re-checks stopFlag promptly instead of
	// blocking for a long time with nothing to do. Confirmed behavior from
	// Arena/IDevice.h::GetBuffer docblock: "A timeout value of 0 ensures the
	// call will not block... GenICam::TimeoutException is thrown" when the
	// timeout elapses with nothing in the output queue.
	const uint64_t kGetBufferTimeoutMs = 200;

	int consecutiveGenericErrors = 0;

	while (!stopFlag.load(std::memory_order_relaxed))
	{
		Arena::IBuffer* pBuffer = nullptr;

		try
		{
			pBuffer = pDevice->GetBuffer(kGetBufferTimeoutMs);
		}
		catch (GenICam::TimeoutException&)
		{
			// No data arrived within this window - not an error, just idle.
			continue;
		}
		catch (GenICam::GenericException& e)
		{
			std::cerr << "[acquisition] GenICam exception: " << e.GetDescription() << std::endl;
			++consecutiveGenericErrors;

			if (!pDevice->IsConnected())
			{
				std::cerr << "[acquisition] Device appears disconnected; "
					<< "waiting for reconnection..." << std::endl;

				const int kMaxReconnectAttempts = 6;   // ~30s total at 5s each
				bool reconnected = false;
				for (int attempt = 0;
					attempt < kMaxReconnectAttempts && !stopFlag.load(std::memory_order_relaxed);
					++attempt)
				{
					if (pDevice->WaitForReconnection(5000))
					{
						reconnected = true;
						break;
					}
				}

				if (reconnected)
				{
					std::cerr << "[acquisition] Device reconnected; resuming acquisition."
						<< std::endl;
					g_stats.reconnectEvents.fetch_add(1, std::memory_order_relaxed);
					consecutiveGenericErrors = 0;
				}
				else
				{
					std::cerr << "[acquisition] Device did not reconnect after "
						<< (kMaxReconnectAttempts * 5) << " seconds; stopping recording."
						<< std::endl;
					stopFlag.store(true, std::memory_order_relaxed);
					break;
				}
			}
			else if (consecutiveGenericErrors > 50)
			{
				std::cerr << "[acquisition] Too many consecutive errors while still "
					<< "reporting connected; stopping recording." << std::endl;
				stopFlag.store(true, std::memory_order_relaxed);
				break;
			}
			continue;
		}

		consecutiveGenericErrors = 0;
		g_stats.buffersAcquired.fetch_add(1, std::memory_order_relaxed);

		// Confirmed from Arena/IBuffer.h::IsIncomplete docblock: signals that
		// GetSizeFilled() does not match the expected 'PayloadSize', usually
		// due to missed packets. We still must requeue it (Arena owns the
		// memory regardless of completeness).
		if (pBuffer->IsIncomplete())
		{
			g_stats.incompleteBuffers.fetch_add(1, std::memory_order_relaxed);
			pDevice->RequeueBuffer(pBuffer);
			g_stats.buffersRequeued.fetch_add(1, std::memory_order_relaxed);
			continue;
		}

		// Take the host arrival stamp FIRST, before any GenApi work below —
		// AsImage()/GetTimestampNs() can lazily parse the payload leader and the
		// debug print can block on stderr, and both would otherwise be charged to
		// this buffer's arrival time.
		const uint64_t hostRecvNs = HostNsSinceStart();

		RecordHeader hdr{};
		hdr.frameId = pBuffer->GetFrameId();
		hdr.payloadSize = static_cast<uint64_t>(pBuffer->GetSizeFilled());
		hdr.hostRecvNs = hostRecvNs;

		// ── The former "fabricated timestamp" site ──────────────────────────
		// V1 wrote 0 here whenever HasImageData() was false, and the offline
		// converter then invented `frame_index / --fps` to fill the hole. V2
		// never leaves a hole: it probes the device clock (ignoring the
		// HasImageData() gate, see TryGetBufferTimestampNs), and if that yields
		// nothing it falls back to the monotonic host arrival stamp taken above
		// — recording WHICH of the two it was, so nothing downstream can mistake
		// an arrival time for a sensor time.
		//
		// Neither number is a per-EVENT timestamp. A dense accumulated frame
		// only asserts "this pixel fired somewhere inside this window"; the
		// sub-window ordering is destroyed inside the camera and is not
		// recoverable here or anywhere else. What this gives the offline side is
		// an honest, measured window boundary instead of a guessed one.
		hdr.deviceTimestampNs = 0;
		if (TryGetBufferTimestampNs(pBuffer, hdr.deviceTimestampNs))
			hdr.timestampSource = kTimestampSourceDevice;
		else
			hdr.timestampSource = kTimestampSourceHost;

		// Step 0 diagnostic (per debugging guide): print payloadType/HasImageData
		// for every buffer, not just the first. Deliberately placed AFTER the
		// timestamp probe so it can report deviceTsOk alongside HasImageData —
		// that pairing is the whole point. Arena documents payloadType==1
		// (BufferPayloadTypeImage) as implying HasImageData()==true; the field
		// log showed "payloadType=1 HasImageData=0", which cannot both be right.
		// Printing HasImageData and the independent AsImage()+GetTimestampNs
		// result side by side settles it from data instead of argument:
		//   HasImageData=1 deviceTsOk=1 -> V1 was leaving a real clock on the floor
		//   HasImageData=0 deviceTsOk=1 -> the HasImageData() gate was the bug
		//   HasImageData=? deviceTsOk=0 -> no device clock exists; host time it is
		// Capped via --debug-buffers [N] (default 20) so it cannot flood a long run.
		if (g_debugBufferCount > 0)
		{
			std::cerr << "[debug] frameId=" << hdr.frameId
				<< " payloadType=" << pBuffer->GetPayloadType()
				<< " HasImageData=" << pBuffer->HasImageData()
				<< " hasChunkData=" << pBuffer->HasChunkData()
				<< " sizeFilled=" << hdr.payloadSize
				<< " deviceTsOk=" << (hdr.timestampSource == kTimestampSourceDevice)
				<< " deviceTsNs=" << hdr.deviceTimestampNs
				<< " hostRecvNs=" << hdr.hostRecvNs
				<< std::endl;
			--g_debugBufferCount;
		}

		const uint8_t* pData = pBuffer->GetData();
		if (pData == nullptr)
		{
			// Documented as possible on a failed/odd payload. memcpy from null is
			// UB, so drop the record rather than crash the recording.
			std::cerr << "[acquisition] buffer " << hdr.frameId
				<< " returned a null data pointer - dropping this record." << std::endl;
			pDevice->RequeueBuffer(pBuffer);
			g_stats.buffersRequeued.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		if (hdr.payloadSize > ring.SlotBytes())
		{
			// Live warning, not just a silent counter — this is exactly the
			// class of bug that previously corrupted an entire recording
			// without any visible sign until the file was inspected offline.
			static std::atomic<int> warnedCount{0};
			if (warnedCount.fetch_add(1, std::memory_order_relaxed) < 5)
			{
				// NOTE: --queue-slots controls how MANY slots exist, not how big
				// each one is, so the old "restart with a larger --queue-slots"
				// advice here was simply wrong and would not have helped anyone
				// who followed it. Slot capacity is derived from the larger of the
				// TL PayloadSize node and the first observed buffer (see main()),
				// so the real remedy is --slot-bytes.
				std::cerr << "[WARNING] buffer " << hdr.frameId << " is " << hdr.payloadSize
					<< " bytes, larger than the ring slot capacity (" << ring.SlotBytes()
					<< " bytes) - IT WILL BE TRUNCATED. Re-run with "
					<< "--slot-bytes " << (hdr.payloadSize * 2)
					<< " to pin a large enough slot size, and investigate why the buffer "
					<< "size grew after the first buffer was measured." << std::endl;
			}
		}
		const bool pushed = ring.TryPush(pData, static_cast<size_t>(hdr.payloadSize), hdr);
		if (!pushed)
			g_stats.queueOverflowDrops.fetch_add(1, std::memory_order_relaxed);

		// ALWAYS requeue immediately, whether or not the push succeeded.
		// Per Arena/IDevice.h::RequeueBuffer docblock: holding buffers risks
		// starving the acquisition engine. This is the single most important
		// rule for a never-block, never-crash recorder.
		pDevice->RequeueBuffer(pBuffer);
		g_stats.buffersRequeued.fetch_add(1, std::memory_order_relaxed);
	}
}

// ============================================================================
// SECTION 7 — Thread B: disk writer
// ============================================================================

void WriterThreadFunc(SpscByteRing& ring, std::ofstream& outFile, std::atomic<bool>& stopFlag)
{
	std::vector<uint8_t> buf;
	size_t size = 0;
	RecordHeader hdr{};

	while (true)
	{
		const bool got = ring.TryPop(buf, size, hdr);
		if (got)
		{
			outFile.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
			outFile.write(reinterpret_cast<const char*>(buf.data()),
				static_cast<std::streamsize>(size));

			// Previously unchecked. std::ofstream does not throw by default, so a
			// full disk, a disconnected USB drive, or a quota hit would silently
			// no-op every subsequent write while the stats line kept happily
			// reporting bytesWritten climbing — the recording would look perfect
			// and be truncated garbage. Fail loudly and stop instead.
			if (!outFile)
			{
				std::cerr << "[writer] FATAL: write failed at record "
					<< g_stats.recordsWritten.load(std::memory_order_relaxed)
					<< " (disk full / device removed / permissions?). Stopping the "
					<< "recording now so the failure is visible instead of silently "
					<< "truncating the file." << std::endl;
				g_writeFailed.store(true, std::memory_order_relaxed);
				stopFlag.store(true, std::memory_order_relaxed);
				break;
			}

			g_stats.bytesWritten.fetch_add(sizeof(hdr) + size, std::memory_order_relaxed);
			g_stats.recordsWritten.fetch_add(1, std::memory_order_relaxed);
		}
		else
		{
			if (stopFlag.load(std::memory_order_relaxed) && ring.Empty())
				break; // fully drained after stop was requested - safe to exit

			// Brief idle backoff. This sleep only ever happens on the
			// CONSUMER side when there is nothing to do; it never affects
			// the producer's non-blocking guarantee.
			std::this_thread::sleep_for(std::chrono::microseconds(200));
		}
	}

	outFile.flush();
	if (!outFile)
	{
		std::cerr << "[writer] FATAL: final flush failed - the output file is incomplete."
			<< std::endl;
		g_writeFailed.store(true, std::memory_order_relaxed);
	}
}

// ============================================================================
// SECTION 8 — Device discovery helper
// ============================================================================

Arena::DeviceInfo FindDevice(Arena::ISystem* pSystem, const std::string& serialFilter)
{
	pSystem->UpdateDevices(2000); // 2s discovery window
	std::vector<Arena::DeviceInfo> devices = pSystem->GetDevices();

	if (devices.empty())
	{
		throw std::runtime_error(
			"No GigE Vision devices discovered. Check PoE link power, IP configuration, "
			"and make sure no other application (e.g. ArenaView) is holding the device open.");
	}

	if (!serialFilter.empty())
	{
		for (auto& info : devices)
		{
			if (std::string(info.SerialNumber().c_str()) == serialFilter)
				return info;
		}
		throw std::runtime_error("No device found matching --serial " + serialFilter);
	}

	if (devices.size() > 1)
	{
		std::ostringstream oss;
		oss << "Multiple devices found; pass --serial <SN> to disambiguate:\n";
		for (auto& info : devices)
		{
			oss << "  model=" << info.ModelName().c_str()
				<< " serial=" << info.SerialNumber().c_str()
				<< " ip=" << info.IpAddressStr().c_str() << "\n";
		}
		throw std::runtime_error(oss.str());
	}

	return devices[0];
}

// ============================================================================
// SECTION 9 — CLI argument parsing
// ============================================================================

struct Args
{
	std::string serial;
	std::string eventFormatName;       // e.g. "EVT3_0" or "EVT2_1"
	std::string eventFormatSizeName;   // e.g. "Bpe16" or "Bpe64" - REQUIRED for recording
	std::string outputPath;
	bool listEventFormats = false;
	bool dumpNodes = false;
	std::string dumpNodesFilter;
	std::string dumpNodesTarget = "device"; // "device" | "stream"
	std::string loadFeaturesPath; // --load-features <file.txt>
	double ercRateLimit = -1.0;   // -1 = don't touch; --erc-rate-limit <Mev/s>
	std::string getIntNodeName; // e.g. "Width" or "Height" — --get-int diagnostic
	std::string getEnumNodeName; // e.g. "TestPattern" — --get-enum diagnostic
	std::string nodeInfoName;    // --node-info <Name>: access-mode + tooltip diagnostic
	std::string setEnumName;     // --set-enum <Name>: direct enum write diagnostic
	std::string setEnumValue;    // --set-enum-value <Value>: paired with setEnumName
	uint64_t durationSeconds = 0; // 0 = run until SIGINT
	size_t numBuffers = 64;
	size_t queueSlots = 256;
	size_t slotBytes = 0;         // --slot-bytes <N>: 0 = auto-size from PayloadSize/first buffer
	uint64_t statsIntervalSeconds = 1;
	int debugBuffers = 0;  // --debug-buffers [N]: print Step-0 per-buffer diagnostic for N buffers
};

void PrintUsage(const char* argv0)
{
	std::cerr <<
		"Usage:\n"
		"  " << argv0 << " --list-event-formats [--serial <SN>]\n"
		"  " << argv0 << " --dump-nodes [--filter <substring>] [--serial <SN>]\n"
		"  " << argv0 << " --get-int <NodeName> [--serial <SN>]\n"
		"  " << argv0 << " --get-enum <NodeName> [--serial <SN>]\n"
		"  " << argv0 << " --output <path.cevt> --event-format <NAME> [options]\n"
		"\n"
		"Required for recording:\n"
		"  --output <path>          Output file path (custom container, see report)\n"
		"  --event-format <NAME>    Exact EventFormat symbolic name, e.g. EVT3_0 or\n"
		"                           EVT2_1 (run --list-event-formats to confirm)\n"
		"\n"
		"Required for recording (both — camera keeps a leftover value from the\n"
		"previous session otherwise, e.g. a stale Bpe16 was seen surviving\n"
		"across runs when this node wasn't explicitly touched):\n"
		"  --event-format-size <NAME>  EventFormatSize symbolic name, e.g. Bpe16/Bpe64\n"
		"\n"
		"Optional:\n"
		"  --serial <SN>            Select a specific camera when multiple are present\n"
		"  --load-features <file>  Apply a saved bias/exposure profile (Arena::FeatureStream)\n"
		"                           BEFORE recording, so one command = one fully specified\n"
		"                           session, independent of whatever was last left on the\n"
		"                           camera (e.g. via ArenaView)\n"
		"  --debug-buffers [N]      Print a per-buffer Step-0 diagnostic (payloadType,\n"
		"                           HasImageData, hasChunkData, sizeFilled, deviceTsOk,\n"
		"                           deviceTsNs, hostRecvNs) for the first N buffers\n"
		"                           (default 20 if the flag is given with no value).\n"
		"                           deviceTsOk is the INDEPENDENT result of attempting\n"
		"                           AsImage()+GetTimestampNs() without gating on\n"
		"                           HasImageData(). Arena documents payloadType==1 as\n"
		"                           implying HasImageData()==true, yet this camera was\n"
		"                           observed reporting payloadType=1 with HasImageData=0 —\n"
		"                           those cannot both be correct, so this flag prints both\n"
		"                           and lets the recording decide. If deviceTsOk=1 you have\n"
		"                           a real device clock; if 0, hostRecvNs is the only real\n"
		"                           time source and every record says so explicitly.\n"
		"  --erc-rate-limit <Mev/s> Set ErcRateLimit (a FLOAT node) before recording. LUCID's\n"
		"                           own recommendation for a 1GigE link is 40. NOTE: there is\n"
		"                           NO node on this camera that counts events ERC silently\n"
		"                           drops - confirmed absent via --dump-nodes --filter Erc.\n"
		"                           Raising the limit is the only real mitigation.\n"
		"  --duration <seconds>     Stop automatically after N seconds (default: run\n"
		"                           until Ctrl+C / SIGINT)\n"
		"  --num-buffers <N>        Arena internal buffer pool depth (default 64, min 1)\n"
		"  --queue-slots <N>        Lock-free ring buffer depth (default 256, min 2)\n"
		"  --slot-bytes <N>         Bytes per ring slot. Default 0 = auto-size from the\n"
		"                           larger of the TL 'PayloadSize' node and the first\n"
		"                           observed buffer, x2. Set this explicitly if you ever\n"
		"                           see a 'IT WILL BE TRUNCATED' warning.\n"
		"  --stats-interval <sec>   Seconds between stats printouts (default 1). Ctrl+C and\n"
		"                           --duration are now honoured within ~100 ms regardless\n"
		"                           of this value.\n"
		"  --list-event-formats     Connect, print EventFormat/EventFormatSize entries, exit\n"
		"  --dump-nodes             Connect, print every node name on the device, exit\n"
		"  --target device|stream   With --dump-nodes, which node map to dump\n"
		"                           (default 'device'; 'stream' = TL Stream node map,\n"
		"                           where PayloadSize/StreamBufferHandlingMode/etc live)\n"
		"  --filter <substring>     With --dump-nodes, only print names containing this\n"
		"  --get-int <NodeName>     Connect, print the current integer value of any named\n"
		"                           node on the main device node map (e.g. Width, Height), exit\n"
		"  --get-enum <NodeName>    Connect, print the current symbolic value of any named\n"
		"                           enum node (e.g. TestPattern, AcquisitionMode), exit\n"
		"  --node-info <NodeName>   Connect, print IsAvailable/IsReadable/IsWritable plus\n"
		"                           ToolTip/Description for any named node (no value read\n"
		"                           or written), exit. Use this to find out WHY a node is\n"
		"                           locked when --get-enum/--set-enum report not-readable/\n"
		"                           not-writable — the camera's own ToolTip text sometimes\n"
		"                           names the gating condition in plain English.\n"
		"  --set-enum <NodeName> --set-enum-value <Value>\n"
		"                           Connect, set one enum node directly (writable-check +\n"
		"                           read-back-verify via the same path recording uses),\n"
		"                           exit. For probing a node without a full recording run.\n"
		"\n"
		"Network tuning (best-effort, non-fatal if unavailable - see console output):\n"
		"  attempts to enable 'StreamPacketResendEnable' and\n"
		"  'StreamAutoNegotiatePacketSize' on the TL Stream node map before recording.\n"
		"  NOTE: these two names are NOT yet confirmed to exist on this camera (unlike\n"
		"  every other node name this program uses) - run\n"
		"    --dump-nodes --target stream --filter Packet\n"
		"  to check the real names first if packet loss is a concern.\n";
}

bool ParseArgs(int argc, char** argv, Args& args, std::string& errorOut)
{
	for (int i = 1; i < argc; ++i)
	{
		std::string a = argv[i];
		auto needValue = [&](const char* flag) -> std::string
		{
			if (i + 1 >= argc)
				throw std::runtime_error(std::string(flag) + " requires a value");
			return argv[++i];
		};

		try
		{
			if (a == "--serial") args.serial = needValue("--serial");
			else if (a == "--event-format") args.eventFormatName = needValue("--event-format");
			else if (a == "--event-format-size") args.eventFormatSizeName = needValue("--event-format-size");
			else if (a == "--output") args.outputPath = needValue("--output");
			else if (a == "--list-event-formats") args.listEventFormats = true;
			else if (a == "--dump-nodes") args.dumpNodes = true;
			else if (a == "--filter") args.dumpNodesFilter = needValue("--filter");
			else if (a == "--target") args.dumpNodesTarget = needValue("--target");
			else if (a == "--load-features") args.loadFeaturesPath = needValue("--load-features");
		else if (a == "--erc-rate-limit") args.ercRateLimit = std::stod(needValue("--erc-rate-limit"));
		else if (a == "--get-int") args.getIntNodeName = needValue("--get-int");
		else if (a == "--get-enum") args.getEnumNodeName = needValue("--get-enum");
		else if (a == "--node-info") args.nodeInfoName = needValue("--node-info");
		else if (a == "--set-enum") args.setEnumName = needValue("--set-enum");
		else if (a == "--set-enum-value") args.setEnumValue = needValue("--set-enum-value");
			else if (a == "--duration") args.durationSeconds = std::stoull(needValue("--duration"));
			else if (a == "--num-buffers") args.numBuffers = std::stoull(needValue("--num-buffers"));
			else if (a == "--queue-slots") args.queueSlots = std::stoull(needValue("--queue-slots"));
			else if (a == "--slot-bytes") args.slotBytes = std::stoull(needValue("--slot-bytes"));
			else if (a == "--stats-interval") args.statsIntervalSeconds = std::stoull(needValue("--stats-interval"));
			else if (a == "--debug-buffers")
			{
				// Optional value: if the next token is a bare (non-flag) integer,
				// consume it; otherwise default to 20 so `--debug-buffers` alone works.
				if (i + 1 < argc && !std::string(argv[i + 1]).empty() && argv[i + 1][0] != '-')
					args.debugBuffers = std::stoi(argv[++i]);
				else
					args.debugBuffers = 20;
			}
			else if (a == "--help" || a == "-h") { errorOut = ""; return false; }
			else { errorOut = "Unknown argument: " + a; return false; }
		}
		catch (std::exception& e)
		{
			errorOut = e.what();
			return false;
		}
	}

	// Validate numeric arguments up front. Previously these went straight into
	// Arena / the ring buffer unchecked: --num-buffers 0 hands 0 to StartStream,
	// --queue-slots 1 throws out of the SpscByteRing constructor with a message
	// that does not name the flag responsible, and --debug-buffers -5 silently
	// disabled the diagnostic the operator had just asked for.
	if (args.numBuffers < 1)
	{
		errorOut = "--num-buffers must be at least 1 (Arena needs a non-empty buffer pool)";
		return false;
	}
	if (args.queueSlots < 2)
	{
		errorOut = "--queue-slots must be at least 2 (the SPSC ring reserves one slot to "
			"distinguish full from empty)";
		return false;
	}
	if (args.debugBuffers < 0)
	{
		errorOut = "--debug-buffers must be >= 0";
		return false;
	}

	if (!args.listEventFormats && !args.dumpNodes && args.getIntNodeName.empty() && args.getEnumNodeName.empty() && args.nodeInfoName.empty() && args.setEnumName.empty())
	{
		if (args.outputPath.empty())
		{
			errorOut = "--output is required (or use --list-event-formats)";
			return false;
		}
		if (args.eventFormatName.empty())
		{
			errorOut = "--event-format is required (run --list-event-formats first)";
			return false;
		}
		if (args.eventFormatSizeName.empty())
		{
			// Previously optional — promoted to required after observing that a
			// leftover EventFormatSize value from a prior session (e.g. a stale
			// Bpe16) survives on the camera when this node is never explicitly
			// touched, and --get-enum after a run with the node unset showed
			// exactly that: a value this recorder never set. Never let the
			// camera's ambient state silently decide this.
			errorOut = "--event-format-size is required (Bpe16 or Bpe64 — run "
				"--list-event-formats first). Do not rely on whatever value the camera "
				"happens to already have; it may be left over from a previous session.";
			return false;
		}
	}

	return true;
}

// ============================================================================
// SECTION 10 — main()
// ============================================================================

int main(int argc, char** argv)
{
	std::signal(SIGINT, SignalHandler);
	std::signal(SIGTERM, SignalHandler);

	Args args;
	std::string parseError;
	if (!ParseArgs(argc, argv, args, parseError))
	{
		if (!parseError.empty())
			std::cerr << "Argument error: " << parseError << "\n\n";
		PrintUsage(argv[0]);
		return parseError.empty() ? 0 : 1;
	}

	Arena::ISystem* pSystem = nullptr;
	Arena::IDevice* pDevice = nullptr;
	std::ofstream outFile;
	RecorderDisconnectCallback disconnectCb;
	bool disconnectCbRegistered = false;

	try
	{
		pSystem = Arena::OpenSystem();
		Arena::DeviceInfo info = FindDevice(pSystem, args.serial);

		std::cout << "Opening device: model=" << info.ModelName().c_str()
			<< " serial=" << info.SerialNumber().c_str()
			<< " ip=" << info.IpAddressStr().c_str() << std::endl;

		pDevice = pSystem->CreateDevice(info);

		GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();
		GenApi::INodeMap* pTLStreamNodeMap = pDevice->GetTLStreamNodeMap();

		// ── Diagnostic early-exit paths ─────────────────────────────
		// NOTE: RegisterDeviceDisconnectCallback is intentionally NOT called
		// before these paths. The callback is only useful during long
		// acquisition (to catch unexpected disconnects mid-recording).
		// For sub-second diagnostic commands, registering and then forgetting
		// to deregister before DestroyDevice causes 'terminate called without
		// an active exception' on Linux (confirmed crash on TRT009S-E).
		// Fix: register the callback AFTER all early exits, just before
		// StartStream() — the only place it actually matters.

		if (args.listEventFormats)
		{
			ListEventFormats(pNodeMap);
			pSystem->DestroyDevice(pDevice);
			Arena::CloseSystem(pSystem);
			return 0;
		}

		if (args.dumpNodes)
		{
			GenApi::INodeMap* pTargetMap = pNodeMap;
			if (args.dumpNodesTarget == "stream")
			{
				pTargetMap = pTLStreamNodeMap;
				std::cout << "Dumping TL STREAM node map (not the main device node map)."
					<< std::endl;
			}
			else if (args.dumpNodesTarget != "device")
			{
				std::cerr << "[warning] --target must be 'device' or 'stream', got '"
					<< args.dumpNodesTarget << "' - defaulting to 'device'." << std::endl;
			}
			DumpAllNodes(pTargetMap, args.dumpNodesFilter);
			pSystem->DestroyDevice(pDevice);
			Arena::CloseSystem(pSystem);
			return 0;
		}

		if (!args.getIntNodeName.empty())
		{
			uint64_t value = ReadStreamIntegerNode(pNodeMap, args.getIntNodeName.c_str());
			std::cout << args.getIntNodeName << " = " << value << std::endl;
			pSystem->DestroyDevice(pDevice);
			Arena::CloseSystem(pSystem);
			return 0;
		}

		if (!args.getEnumNodeName.empty())
		{
			PrintEnumCurrentValue(pNodeMap, args.getEnumNodeName.c_str());
			pSystem->DestroyDevice(pDevice);
			Arena::CloseSystem(pSystem);
			return 0;
		}

		// --node-info <Name>: dump access mode + tooltip/description without
		// reading or writing a value. Added specifically to investigate nodes
		// that are present in --dump-nodes but return errors on --get-enum/
		// --set-enum (e.g. AcquisitionAccumulationMode reporting "not
		// readable"/"not writable") — this tells you WHY (IsAvailable/
		// IsReadable/IsWritable) and often the camera's own ToolTip text names
		// the gating condition in plain English.
		if (!args.nodeInfoName.empty())
		{
			GenApi::CNodePtr pNode = pNodeMap->GetNode(args.nodeInfoName.c_str());
			if (!pNode.IsValid())
			{
				std::cout << args.nodeInfoName << ": not found" << std::endl;
			}
			else
			{
				std::cout << std::boolalpha;
				std::cout << args.nodeInfoName << ":\n"
					<< "  IsAvailable = " << GenApi::IsAvailable(pNode) << "\n"
					<< "  IsReadable  = " << GenApi::IsReadable(pNode) << "\n"
					<< "  IsWritable  = " << GenApi::IsWritable(pNode) << "\n"
					<< "  ToolTip     = " << pNode->GetToolTip() << "\n"
					<< "  Description = " << pNode->GetDescription() << "\n";

				// Print the actual VALUE too when the node is readable. This was
				// specified in an earlier session and then never landed, which is
				// why "check DeviceFirmwareVersion against LUCID's changelog" kept
				// being listed as a next step that could not actually be performed
				// — --node-info printed access flags but no value, and --get-enum
				// only handles enums. GenApi::CValuePtr is the generic interface
				// that every node type (string, integer, float, enum, boolean)
				// implements, so one ToString() covers all of them.
				if (GenApi::IsReadable(pNode))
				{
					try
					{
						GenApi::CValuePtr pValue = pNode;
						if (pValue.IsValid())
							std::cout << "  Value       = " << pValue->ToString().c_str() << "\n";
					}
					catch (GenICam::GenericException& e)
					{
						std::cout << "  Value       = <read failed: " << e.GetDescription() << ">\n";
					}
				}
			}
			pSystem->DestroyDevice(pDevice);
			Arena::CloseSystem(pSystem);
			return 0;
		}

		// --set-enum <Name> --set-enum-value <Value>: set a single enum node
		// directly, without recording, reusing ApplyEnumIfPresent's writable-
		// check + read-back-verify logic. Useful for probing whether a node
		// that looks locked (IsWritable=false) actually accepts a write when
		// tried directly, and for one-off experiments (e.g. trying
		// AcquisitionAccumulationMode=EventBased) without a full recording run.
		if (!args.setEnumName.empty())
		{
			if (args.setEnumValue.empty())
			{
				std::cerr << "Error: --set-enum requires --set-enum-value to also be specified." << std::endl;
			}
			else
			{
				ApplyEnumIfPresent(pNodeMap, args.setEnumName.c_str(), args.setEnumValue, /*required=*/true);
			}
			pSystem->DestroyDevice(pDevice);
			Arena::CloseSystem(pSystem);
			return 0;
		}

		// ── Recording path: register disconnect callback NOW ─────────
		// Only reached when actually recording. Deregistered explicitly
		// before DestroyDevice on both the success and failure paths below.
		pSystem->RegisterDeviceDisconnectCallback(pDevice, &disconnectCb);
		disconnectCbRegistered = true;

		LoadFeaturesIfRequested(pNodeMap, args.loadFeaturesPath);

		// Both required (ParseArgs enforces this) and both now confirm-read-back
		// after SetIntValue — see ApplyEnumIfPresent. This is the camera's ONE
		// real, confirmed decode path; there is no XYPT request anymore (see
		// the removed-TryApplyOutputFormat comment above SECTION 9/10 for why).
		ApplyEnumIfPresent(pNodeMap, "EventFormat", args.eventFormatName, true);
		ApplyEnumIfPresent(pNodeMap, "EventFormatSize", args.eventFormatSizeName, true);

		g_debugBufferCount = args.debugBuffers;

		// ── Timebase setup — the actual cure for the fabricated timestamp ────
		// The camera cannot be coaxed into sparse output (AcquisitionAccumulationMode
		// is firmware-locked, IsAvailable=false — exhaustively verified), so every
		// payload is a dense accumulated frame covering one accumulation window.
		// The window LENGTH is not a mystery, though: it is whatever
		// AcquisitionFrameRate / AcquisitionFrameTime say, and both are ordinary
		// readable nodes on this device's AcquisitionControl. Reading them here and
		// storing them in the file header is what lets the offline converter stop
		// asking the operator to type "--fps 30" and hope.
		double frameRateHz = ReadFloatNode(pNodeMap, "AcquisitionFrameRate");
		uint64_t frameTimeUs = 0;
		const bool haveFrameTime = TryReadIntegerNode(pNodeMap, "AcquisitionFrameTime", frameTimeUs);

		// Cross-check the two against each other rather than trusting either
		// alone: they describe the same quantity, so a disagreement means one of
		// them is stale or in different units, and silently picking one would
		// reintroduce exactly the class of bug being fixed.
		if (frameRateHz > 0 && haveFrameTime && frameTimeUs > 0)
		{
			const double impliedHz = 1e6 / static_cast<double>(frameTimeUs);
			const double relErr = std::abs(impliedHz - frameRateHz) / frameRateHz;
			if (relErr > 0.05)
			{
				std::cerr << "[timebase][warning] AcquisitionFrameRate (" << frameRateHz
					<< " Hz) and AcquisitionFrameTime (" << frameTimeUs << " us => "
					<< impliedHz << " Hz) disagree by " << (relErr * 100.0) << "%. Both are "
					<< "being recorded verbatim; the offline converter will prefer "
					<< "AcquisitionFrameTime and flag the discrepancy rather than average "
					<< "them." << std::endl;
			}
		}

		if (frameRateHz <= 0 && !haveFrameTime)
		{
			std::cerr << "[timebase][warning] Neither AcquisitionFrameRate nor "
				<< "AcquisitionFrameTime could be read. The accumulation-window length "
				<< "will be UNKNOWN in this recording, and cevt_to_events.py will refuse "
				<< "to synthesise timestamps unless you pass --fps explicitly. That is "
				<< "intentional: a guessed window silently corrupts every rate and "
				<< "timing metric derived from this file." << std::endl;
		}
		else
		{
			std::cout << "[timebase] AcquisitionFrameRate = "
				<< (frameRateHz > 0 ? std::to_string(frameRateHz) + " Hz" : std::string("unreadable"))
				<< ", AcquisitionFrameTime = "
				<< (haveFrameTime ? std::to_string(frameTimeUs) + " us" : std::string("unreadable"))
				<< std::endl;
		}

		// Reset the device's free-running Timestamp counter so device time and
		// host time share an origin. Best-effort: a camera that refuses this is
		// still recordable, we just lose the ability to correlate the two clocks
		// after the fact — which is recorded in FileHeader::flags rather than
		// assumed either way.
		const bool timestampResetOk = TryExecuteCommand(pNodeMap, "TimestampReset");

		SetStreamBufferHandlingModeOldestFirst(pTLStreamNodeMap);

		uint64_t payloadSizeFromNode = GetStreamPayloadSize(pTLStreamNodeMap);
		if (payloadSizeFromNode == 0)
		{
			std::cerr << "[warning] Could not read 'PayloadSize' from the stream node map; "
				<< "will size the ring buffer from the first captured buffer instead." << std::endl;
		}

		// Network tuning - see TryEnableBoolean's doc comment for the honesty
		// caveat on these two node names (not yet confirmed against this
		// camera's TL Stream node map). Failure here is non-fatal by design.
		TryEnableBoolean(pTLStreamNodeMap, "StreamPacketResendEnable", true);
		TryEnableBoolean(pTLStreamNodeMap, "StreamAutoNegotiatePacketSize", true);

		// ERC (Event Rate Control) status - always printed, whether or not
		// --erc-rate-limit is used, so every recording has a paper trail of
		// what limit was active. IMPORTANT CAVEAT (confirmed via
		// --dump-nodes --filter Erc against the real TRT009S-E): there is
		// NO node that counts events ERC silently discards. If the true
		// event rate exceeds ErcRateLimit, data is lost with zero visibility
		// from host-side counters (gvspMissedPackets/incomplete/queueDrops
		// all stay 0, because from the host's perspective those events never
		// existed). Raising the limit is the only real mitigation available.
		if (args.ercRateLimit > 0)
			TrySetFloat(pNodeMap, "ErcRateLimit", args.ercRateLimit);
		{
			double curLimit = ReadFloatNode(pNodeMap, "ErcRateLimit");
			std::cout << "[erc] Current ErcRateLimit = "
				<< (curLimit >= 0 ? std::to_string(curLimit) + " Mev/s" : "unreadable")
				<< "  (LUCID's own recommendation for 1GigE is 40; events above this "
				<< "limit are dropped INSIDE the camera with no host-visible counter)"
				<< std::endl;
		}

		outFile.open(args.outputPath, std::ios::binary | std::ios::trunc);
		if (!outFile.is_open())
			throw std::runtime_error("Could not open output file: " + args.outputPath);

		// Latch the host monotonic origin IMMEDIATELY before StartStream, and the
		// matching wall-clock + device-clock readings alongside it, so all three
		// timebases in the file share one defined instant.
		g_hostT0 = std::chrono::steady_clock::now();
		const uint64_t hostUnixEpochNsAtStart = static_cast<uint64_t>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::system_clock::now().time_since_epoch()).count());
		uint64_t deviceTsAtStartNs = 0;
		const bool haveDeviceTsAtStart = TryLatchDeviceTimestampNs(pNodeMap, deviceTsAtStartNs);

		pDevice->StartStream(args.numBuffers);
		std::cout << "Stream started. Recording to " << args.outputPath << " ..." << std::endl;

		// ---- One-time setup: capture the first buffer to learn geometry and
		// pixel format, write the file header, then hand off to the
		// steady-state threads for everything after it. ----
		Arena::IBuffer* pFirst = pDevice->GetBuffer(5000);
		if (pFirst == nullptr)
			throw std::runtime_error(
				"GetBuffer() returned null for the first buffer. Nothing to record.");
		const uint64_t firstHostRecvNs = HostNsSinceStart();
		DiagnoseFirstBuffer(pFirst);

		FileHeader fileHeader{};
		std::memcpy(fileHeader.magic, "CAROEVT2", 8);
		fileHeader.recordHeaderSize = static_cast<uint32_t>(sizeof(RecordHeader));
		fileHeader.acquisitionFrameRateHz = frameRateHz > 0 ? frameRateHz : 0.0;
		fileHeader.acquisitionFrameTimeUs = haveFrameTime ? frameTimeUs : 0;
		fileHeader.hostUnixEpochNsAtStart = hostUnixEpochNsAtStart;
		fileHeader.deviceTimestampAtStartNs = haveDeviceTsAtStart ? deviceTsAtStartNs : 0;
		fileHeader.flags = 0;
		fileHeader.reserved = 0;
		if (timestampResetOk)
			fileHeader.flags |= kFileFlagTimestampResetOk;
		if (frameRateHz > 0 || haveFrameTime)
			fileHeader.flags |= kFileFlagFrameRateKnown;

		// Geometry: prefer the image interface when the cast works. Note this no
		// longer gates on HasImageData() either — same reasoning as
		// TryGetBufferTimestampNs. If the cast genuinely fails we fall back to
		// payload type + zero geometry, exactly as before.
		Arena::IImage* pFirstImage = nullptr;
		try { pFirstImage = pFirst->AsImage(); }
		catch (GenICam::GenericException&) { pFirstImage = nullptr; }

		if (pFirstImage != nullptr)
		{
			fileHeader.pixelFormat = pFirstImage->GetPixelFormat();
			fileHeader.bitsPerPixel = static_cast<uint32_t>(pFirstImage->GetBitsPerPixel());
			fileHeader.width = static_cast<uint32_t>(pFirstImage->GetWidth());
			fileHeader.height = static_cast<uint32_t>(pFirstImage->GetHeight());
		}
		else
		{
			// Payload is not image-typed (see diagnostic line just printed).
			// Rather than crash, record what we do know (payload type) and
			// leave geometry as 0/unknown - the raw bytes are still captured
			// losslessly for offline analysis in Phase 5.
			std::cerr << "[warning] First buffer could not be cast to IImage (payloadType="
				<< pFirst->GetPayloadType() << "). Recording payload bytes verbatim; "
				<< "geometry fields in the file header will be 0 (unknown)." << std::endl;
			fileHeader.pixelFormat = static_cast<uint64_t>(pFirst->GetPayloadType());
			fileHeader.bitsPerPixel = 0;
			fileHeader.width = 0;
			fileHeader.height = 0;
		}

		RecordHeader firstHdr{};
		firstHdr.frameId = pFirst->GetFrameId();
		firstHdr.payloadSize = static_cast<uint64_t>(pFirst->GetSizeFilled());
		firstHdr.hostRecvNs = firstHostRecvNs;
		firstHdr.deviceTimestampNs = 0;
		if (TryGetBufferTimestampNs(pFirst, firstHdr.deviceTimestampNs))
		{
			firstHdr.timestampSource = kTimestampSourceDevice;
			fileHeader.flags |= kFileFlagDeviceTsAvailable;
		}
		else
		{
			firstHdr.timestampSource = kTimestampSourceHost;
		}

		// FileHeader is only written now, after the flags it carries are actually
		// known. Writing it before probing the first buffer (as V1 did) would have
		// forced kFileFlagDeviceTsAvailable to be guessed.
		outFile.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
		outFile.write(reinterpret_cast<const char*>(&firstHdr), sizeof(firstHdr));

		const uint8_t* pFirstData = pFirst->GetData();
		if (pFirstData == nullptr)
			throw std::runtime_error("First buffer has a null data pointer; refusing to record.");
		outFile.write(reinterpret_cast<const char*>(pFirstData),
			static_cast<std::streamsize>(firstHdr.payloadSize));
		if (!outFile)
			throw std::runtime_error(
				"Failed writing the file header / first record to " + args.outputPath +
				" (disk full or not writable?).");

		g_stats.bytesWritten.fetch_add(sizeof(fileHeader) + sizeof(firstHdr) + firstHdr.payloadSize,
			std::memory_order_relaxed);
		g_stats.recordsWritten.fetch_add(1, std::memory_order_relaxed);
		g_stats.buffersAcquired.fetch_add(1, std::memory_order_relaxed);

		pDevice->RequeueBuffer(pFirst);
		g_stats.buffersRequeued.fetch_add(1, std::memory_order_relaxed);
		pFirst = nullptr;      // requeued: any further use is undefined behaviour
		pFirstImage = nullptr; // IImage view of a requeued buffer is equally dead

		std::cout << "Detected stream geometry: " << fileHeader.width << "x" << fileHeader.height
			<< ", pixelFormat=0x" << std::hex << fileHeader.pixelFormat << std::dec
			<< ", bitsPerPixel=" << fileHeader.bitsPerPixel
			<< ", firstBufferTimestampSource="
			<< (firstHdr.timestampSource == kTimestampSourceDevice ? "device" : "host")
			<< std::endl;

		// ---- Ring buffer sizing (FIXED): take whichever estimate is LARGER —
		// the TL stream 'PayloadSize' node, or the size of the buffer we just
		// actually observed. Earlier versions trusted only the node value and
		// silently truncated every subsequent buffer to a too-small slot size
		// whenever the node under-reported the true buffer size (confirmed to
		// happen on this camera: node reported far less than the real ~900KB
		// buffers, causing every non-first record to be cut down to a fixed,
		// wrong size). Taking the max of both, with a safety margin, means a
		// single undersized node reading can no longer silently corrupt the whole
		// recording.
		const uint64_t candidateFromNode = payloadSizeFromNode;
		const uint64_t candidateFromFirstBuffer = firstHdr.payloadSize;
		uint64_t bestEstimate = (std::max)(candidateFromNode, candidateFromFirstBuffer);

		// Guard the degenerate case the old code walked straight into: if the
		// PayloadSize node is unreadable (0) AND the first buffer happens to be
		// empty (0 bytes), bestEstimate is 0, slotCapacity becomes 0, and EVERY
		// subsequent buffer is silently truncated to nothing — a recording that
		// completes "successfully" and contains no data at all.
		if (bestEstimate == 0)
		{
			bestEstimate = 1280ull * 720ull; // this sensor's dense frame size
			std::cerr << "[warning] Both the 'PayloadSize' node and the first buffer "
				<< "reported 0 bytes. Falling back to a " << bestEstimate
				<< "-byte slot estimate so the ring is not created with zero-size slots "
				<< "(which would truncate every record to nothing). Use --slot-bytes to "
				<< "override." << std::endl;
		}

		size_t slotCapacity = args.slotBytes > 0
			? args.slotBytes
			: static_cast<size_t>(bestEstimate) * 2; // safety margin

		// The old form of this test was `candidateFromFirstBuffer > candidateFromNode * 2`,
		// which is trivially true whenever the node reads 0 — so the scary
		// "badly under-estimated" warning fired on every run where the node was
		// simply absent, training the operator to ignore it. Only compare when
		// there is actually a node reading to compare against.
		if (candidateFromNode > 0 && candidateFromFirstBuffer > candidateFromNode * 2)
		{
			std::cerr << "[warning] TL stream 'PayloadSize' node (" << candidateFromNode
				<< " bytes) badly under-estimated the real first buffer size ("
				<< candidateFromFirstBuffer << " bytes). Sizing the ring buffer from the "
				<< "observed buffer instead - this is exactly the mismatch that caused "
				<< "silent truncation in earlier runs." << std::endl;
		}

		const double footprintMiB =
			static_cast<double>(args.queueSlots) * static_cast<double>(slotCapacity) / (1024.0 * 1024.0);
		std::cout << "Ring buffer: " << args.queueSlots << " slots x " << slotCapacity
			<< " bytes = " << footprintMiB << " MiB resident." << std::endl;

		// ---- Steady state: spin up the two worker threads ----
		SpscByteRing ring(args.queueSlots, slotCapacity);

		std::thread acqThread(AcquisitionThreadFunc, pDevice, std::ref(ring), std::ref(g_stopRequested));
		std::thread writerThread(WriterThreadFunc, std::ref(ring), std::ref(outFile), std::ref(g_stopRequested));

		const auto startTime = std::chrono::steady_clock::now();
		uint64_t lastBytesWritten = 0;
		auto lastStatsTime = startTime;

		while (!g_stopRequested.load(std::memory_order_relaxed))
		{
			// Sleep in short slices instead of one long `sleep_for(statsInterval)`.
			// The old form made both Ctrl+C and --duration only take effect at the
			// NEXT stats tick: `--stats-interval 60 --duration 10` recorded 60
			// seconds, and SIGINT could hang the process for a full interval. The
			// slice length is capped so responsiveness never depends on how the
			// user configured logging.
			const auto kSlice = std::chrono::milliseconds(100);
			const auto statsPeriod = std::chrono::seconds(args.statsIntervalSeconds);
			while (!g_stopRequested.load(std::memory_order_relaxed))
			{
				std::this_thread::sleep_for(kSlice);

				const auto now = std::chrono::steady_clock::now();

				if (args.durationSeconds > 0)
				{
					const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
						now - startTime).count();
					if (static_cast<uint64_t>(elapsed) >= args.durationSeconds)
					{
						g_stopRequested.store(true, std::memory_order_relaxed);
						break;
					}
				}

				if (now - lastStatsTime >= statsPeriod)
					break;
			}
			if (g_stopRequested.load(std::memory_order_relaxed))
				break;

			const auto nowStats = std::chrono::steady_clock::now();
			// Measure the REAL elapsed interval rather than assuming it equalled
			// statsIntervalSeconds. That assumption also divided by zero whenever
			// --stats-interval 0 was passed, producing inf/nan throughput.
			const double elapsedSec = std::chrono::duration<double>(nowStats - lastStatsTime).count();
			lastStatsTime = nowStats;

			const uint64_t bytesNow = g_stats.bytesWritten.load(std::memory_order_relaxed);
			const double throughputMBs = elapsedSec > 0.0
				? static_cast<double>(bytesNow - lastBytesWritten) / (1024.0 * 1024.0) / elapsedSec
				: 0.0;
			lastBytesWritten = bytesNow;

			const uint64_t missedPackets = ReadStreamIntegerNode(pTLStreamNodeMap, "StreamMissedPacketCount");
			const uint64_t lostFrames = ReadStreamIntegerNode(pTLStreamNodeMap, "StreamLostFrameCount");

			std::cout << "[stats] acquired=" << g_stats.buffersAcquired.load(std::memory_order_relaxed)
				<< " requeued=" << g_stats.buffersRequeued.load(std::memory_order_relaxed)
				<< " incomplete=" << g_stats.incompleteBuffers.load(std::memory_order_relaxed)
				<< " queueDrops=" << g_stats.queueOverflowDrops.load(std::memory_order_relaxed)
				<< " truncated=" << ring.TruncationCount()
				<< " reconnects=" << g_stats.reconnectEvents.load(std::memory_order_relaxed)
				<< " records=" << g_stats.recordsWritten.load(std::memory_order_relaxed)
				<< " bytesWritten=" << bytesNow
				<< " throughput=" << throughputMBs << "MB/s"
				<< " gvspMissedPackets=" << missedPackets
				<< " gvspLostFrames=" << lostFrames
				<< std::endl;
		}

		std::cout << "Stopping..." << std::endl;
		// Join the acquisition thread FIRST: it is the producer, and the writer's
		// exit condition is "stop requested AND ring empty". Joining the writer
		// first would deadlock if the producer were still pushing.
		acqThread.join();
		writerThread.join();

		pDevice->StopStream();
		outFile.flush();
		outFile.close();
		if (!outFile && !g_writeFailed.load(std::memory_order_relaxed))
		{
			std::cerr << "[fatal] Closing " << args.outputPath << " reported an error - "
				<< "the recording may be incomplete." << std::endl;
			g_writeFailed.store(true, std::memory_order_relaxed);
		}

		const uint64_t truncatedCount = ring.TruncationCount();
		const uint64_t droppedCount = g_stats.queueOverflowDrops.load();
		const uint64_t incompleteCount = g_stats.incompleteBuffers.load();

		std::cout << "Final stats: acquired=" << g_stats.buffersAcquired.load()
			<< " requeued=" << g_stats.buffersRequeued.load()
			<< " incomplete=" << incompleteCount
			<< " queueDrops=" << droppedCount
			<< " truncated=" << truncatedCount
			<< " reconnects=" << g_stats.reconnectEvents.load()
			<< " records=" << g_stats.recordsWritten.load()
			<< " bytesWritten=" << g_stats.bytesWritten.load()
			<< std::endl;

		// Make data loss visible in the EXIT CODE, not only in stdout. A batch
		// script recording several sites had no way to distinguish a clean run
		// from one that silently truncated or dropped buffers, because this
		// program returned 0 either way.
		if (truncatedCount > 0)
			std::cerr << "[fatal] " << truncatedCount << " buffer(s) were TRUNCATED to the ring "
				<< "slot size. This recording is corrupt; re-run with --slot-bytes larger "
				<< "than the largest observed payload." << std::endl;
		if (droppedCount > 0)
			std::cerr << "[warning] " << droppedCount << " buffer(s) were dropped because the "
				<< "ring was full (disk too slow). Increase --queue-slots or write to a "
				<< "faster disk." << std::endl;

		// Symmetric with the failure-path cleanup below: deregister the
		// disconnect callback before destroying the device. DestroyDevice's
		// docblock (Arena/ISystem.h) only promises to close an open stream,
		// deallocate node maps, and close the message/control channels - it
		// does NOT explicitly promise to deregister disconnect callbacks, so
		// this is done explicitly here rather than assumed.
		if (disconnectCbRegistered)
		{
			pSystem->DeregisterDeviceDisconnectCallback(&disconnectCb);
			disconnectCbRegistered = false;
		}
		pSystem->DestroyDevice(pDevice);
		Arena::CloseSystem(pSystem);
		return (g_writeFailed.load(std::memory_order_relaxed) || truncatedCount > 0) ? 2 : 0;
	}
	catch (GenICam::GenericException& e)
	{
		std::cerr << "[fatal] GenICam exception: " << e.GetDescription() << std::endl;
	}
	catch (std::exception& e)
	{
		std::cerr << "[fatal] " << e.what() << std::endl;
	}

	// Best-effort cleanup on any failure path above.
	if (outFile.is_open())
		outFile.close();
	if (pDevice != nullptr && pSystem != nullptr)
	{
		try
		{
			if (disconnectCbRegistered)
				pSystem->DeregisterDeviceDisconnectCallback(&disconnectCb);
			pSystem->DestroyDevice(pDevice);
		}
		catch (...) {}
	}
	if (pSystem != nullptr)
	{
		try { Arena::CloseSystem(pSystem); }
		catch (...) {}
	}
	return 1;
}
