//! Optional recording of the desktop stream to a local MP4 file.
//!
//! Disabled unless `UU_RECORD_DIR` is set: with the variable unset (or empty) `Recorder::start`
//! returns `None` and the desktop stream behaves exactly as before. When it is set, each desktop
//! session is recorded into that directory as `uu-remote-<UTC timestamp>-<codec>.mp4`, with a
//! numeric suffix when several segments land in the same second or a settings change (codec /
//! frame rate) splits a session into several files.
//!
//! The recorder taps the already-encoded access units in `video::stream_desktop`, after the
//! encoder's dependency gate, so a recording always starts on an independent frame (SPS/PPS —
//! and VPS for HEVC — are repeated in-band at every IDR). A second FFmpeg process muxes the
//! elementary stream into MP4 with `-c:v copy`: no decoding and no re-encoding, so recording
//! costs one extra copy of the compressed stream and no extra encoder load. HEVC is stored
//! with the `hvc1` tag, which is the variant Apple players accept; some other players (notably
//! older Windows Media Player builds) cannot play HEVC in MP4 at all — use VLC, or force
//! H.264 for the session with `UU_VIDEO_CODEC=h264`.
//!
//! The elementary stream carries no timestamps, so the muxer stamps frames at the session's
//! nominal (negotiated) frame rate: a capture stall is not preserved as a pause in the
//! recording, and a long-stalled session produces a slightly shorter file than wall-clock time.
//!
//! Recording is best-effort by design: a missing/unwritable directory, a failed FFmpeg start or
//! a write error only logs a warning and stops the recording. It never propagates an error into
//! the desktop stream. Only the real desktop stream is recorded; the synthetic black frames
//! sent while a Portal desktop is locked (`locked_video`) never pass through here. Audio is
//! not recorded (video only). Log lines carry file names, frame counts and sizes only — never
//! frame contents (see AGENTS.md ground rule 1).

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::io::AsyncWriteExt;
use tokio::process::{Child, ChildStdin};
use tokio::sync::mpsc;

use crate::capture_setting::VideoCodec;

/// Frames buffered between the desktop stream and the muxer task. The stream never blocks on
/// recording: when this queue is full (disk slower than the encoder) the recording is given
/// up for the rest of the session instead of back-pressuring the remote desktop — dropping a
/// P-frame would corrupt every following frame until the next IDR anyway.
const QUEUE_FRAMES: usize = 32;

struct RecordFrame {
    codec: VideoCodec,
    fps: u32,
    independent: bool,
    data: Vec<u8>,
}

/// Handle held by `video::stream_desktop`. Dropping it closes the queue, which tells the
/// muxer task to close FFmpeg's stdin, wait for the MP4 index (`moov`) to be written and exit:
/// a session that ends normally therefore always leaves a complete, playable file.
pub struct Recorder {
    tx: Option<mpsc::Sender<RecordFrame>>,
    disabled: Arc<AtomicBool>,
    warned_full: AtomicBool,
}

impl Recorder {
    /// `None` (recording off) unless `UU_RECORD_DIR` names a directory we can create.
    pub fn start() -> Option<Recorder> {
        let dir = record_dir(std::env::var("UU_RECORD_DIR").ok().as_deref())?;
        if let Err(error) = std::fs::create_dir_all(&dir) {
            log::warn!("Desktop recording disabled: cannot create UU_RECORD_DIR {}: {error}", dir.display());
            return None;
        }
        let (tx, rx) = mpsc::channel(QUEUE_FRAMES);
        let disabled = Arc::new(AtomicBool::new(false));
        log::info!("Desktop recording enabled: dir={}", dir.display());
        tokio::spawn(run(dir, rx, disabled.clone()));
        Some(Recorder { tx: Some(tx), disabled, warned_full: AtomicBool::new(false) })
    }

    /// Offer one encoded access unit. Non-blocking; a full queue disables further recording
    /// for this session (see QUEUE_FRAMES) instead of slowing the stream down.
    pub fn record_frame(&self, codec: VideoCodec, fps: u32, independent: bool, data: &[u8]) {
        if self.disabled.load(Ordering::Relaxed) { return; }
        let Some(tx) = &self.tx else { return };
        let frame = RecordFrame { codec, fps, independent, data: data.to_vec() };
        match tx.try_send(frame) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                if !self.warned_full.swap(true, Ordering::Relaxed) {
                    log::warn!("Desktop recording stopped: muxer is slower than the stream (queue full)");
                }
                self.disabled.store(true, Ordering::Relaxed);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => self.disabled.store(true, Ordering::Relaxed),
        }
    }
}

/// `UU_RECORD_DIR` parsing, kept pure for tests: unset, empty and whitespace-only mean "off".
fn record_dir(value: Option<&str>) -> Option<PathBuf> {
    let value = value?.trim();
    (!value.is_empty()).then(|| PathBuf::from(value))
}

fn input_format(codec: VideoCodec) -> &'static str {
    match codec { VideoCodec::H264 => "h264", VideoCodec::H265 => "hevc" }
}

fn codec_label(codec: VideoCodec) -> &'static str {
    match codec { VideoCodec::H264 => "h264", VideoCodec::H265 => "h265" }
}

/// FFmpeg arguments for one recording segment, kept pure for tests. Input is the raw Annex B
/// elementary stream on stdin at the nominal frame rate; output is a stream-copy MP4. HEVC
/// gets the `hvc1` tag (QuickTime/Apple-compatible); `+faststart` moves the index to the
/// front at close so the file plays/downloads progressively.
fn muxer_args(codec: VideoCodec, fps: u32, path: &Path) -> Vec<String> {
    let mut args: Vec<String> = [
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-framerate", &fps.to_string(), "-f", input_format(codec), "-i", "pipe:0",
        "-an", "-c:v", "copy",
    ].into_iter().map(str::to_owned).collect();
    if codec == VideoCodec::H265 { args.extend(["-tag:v".into(), "hvc1".into()]); }
    args.extend(["-movflags".into(), "+faststart".into(), "-f".into(), "mp4".into(),
        path.to_string_lossy().into_owned()]);
    args
}

/// UTC `YYYYMMDD-HHMMSS` for a Unix timestamp, without a date crate. The civil-from-days
/// conversion is Howard Hinnant's algorithm (public domain).
fn timestamp_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}{month:02}{day:02}-{:02}{:02}{:02}", tod / 3_600, tod / 60 % 60, tod % 60)
}

fn now_stamp() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    timestamp_utc(secs)
}

/// First free `uu-remote-<stamp>-<codec>[-n].mp4` inside `dir`.
fn segment_path(dir: &Path, codec: VideoCodec, stamp: &str) -> PathBuf {
    let base = format!("uu-remote-{stamp}-{}", codec_label(codec));
    let mut candidate = dir.join(format!("{base}.mp4"));
    let mut suffix = 2;
    while candidate.exists() {
        candidate = dir.join(format!("{base}-{suffix}.mp4"));
        suffix += 1;
    }
    candidate
}

struct Muxer {
    child: Child,
    stdin: Option<ChildStdin>,
    codec: VideoCodec,
    fps: u32,
    path: PathBuf,
    frames: u64,
    bytes: u64,
}

impl Muxer {
    async fn start(dir: &Path, codec: VideoCodec, fps: u32) -> Result<Muxer, String> {
        let path = segment_path(dir, codec, &now_stamp());
        let mut command = crate::video::ffmpeg_command();
        command.args(muxer_args(codec, fps, &path))
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        let mut child = command.spawn().map_err(|e| format!("Start recording FFmpeg: {e}"))?;
        let stdin = child.stdin.take().ok_or("Recording FFmpeg stdin missing")?;
        log::info!("Desktop recording segment started: file={} codec={} fps={}",
            path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), codec_label(codec), fps);
        Ok(Muxer { child, stdin: Some(stdin), codec, fps, path, frames: 0, bytes: 0 })
    }

    async fn write(&mut self, data: &[u8]) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("Recording FFmpeg stdin closed")?;
        stdin.write_all(data).await.map_err(|e| format!("Recording write: {e}"))?;
        self.frames += 1;
        self.bytes += data.len() as u64;
        Ok(())
    }

    /// Close stdin so FFmpeg writes the MP4 index, then wait (bounded) for it to exit.
    async fn finish(mut self) {
        drop(self.stdin.take());
        match tokio::time::timeout(Duration::from_secs(10), self.child.wait()).await {
            Ok(Ok(status)) if status.success() => log::info!(
                "Desktop recording saved: file={} frames={} bytes={}",
                self.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                self.frames, self.bytes),
            Ok(Ok(status)) => log::warn!(
                "Desktop recording FFmpeg exited with {status}: file={} frames={} (file may be incomplete)",
                self.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), self.frames),
            Ok(Err(error)) => log::warn!("Desktop recording FFmpeg wait failed: {error}"),
            Err(_) => {
                log::warn!("Desktop recording FFmpeg did not finish in 10s; killing it (file may be incomplete)");
                let _ = self.child.kill().await;
            }
        }
    }
}

/// Muxer task: owns every recording segment of one desktop session. A codec/fps change ends
/// the current segment at the next independent frame and starts a new file; frames before the
/// first independent frame of a segment are skipped so every file is self-contained.
async fn run(dir: PathBuf, mut rx: mpsc::Receiver<RecordFrame>, disabled: Arc<AtomicBool>) {
    // start() already created the directory; repeat it here so the task is self-contained
    // (and a directory deleted mid-session is recreated for the next segment).
    if let Err(error) = std::fs::create_dir_all(&dir) {
        log::warn!("Desktop recording disabled: cannot create UU_RECORD_DIR {}: {error}", dir.display());
        disabled.store(true, Ordering::Relaxed);
        return;
    }
    let mut muxer: Option<Muxer> = None;
    while let Some(frame) = rx.recv().await {
        match muxer.as_ref().map(|current| (current.codec, current.fps)) {
            Some((codec, fps)) if codec == frame.codec && fps == frame.fps => {
                if let Err(error) = muxer.as_mut().expect("muxer checked above").write(&frame.data).await {
                    log::warn!("Desktop recording stopped: {error}");
                    if let Some(muxer) = muxer.take() { muxer.finish().await; }
                    disabled.store(true, Ordering::Relaxed);
                    break;
                }
                continue;
            }
            Some(_) => {
                // Settings changed mid-session: only an independent frame may open the next
                // segment; finish the old file now and drop dependent frames until one arrives.
                if let Some(muxer) = muxer.take() { muxer.finish().await; }
                if !frame.independent { continue; }
            }
            None if !frame.independent => continue, // A recording must start at an IDR.
            None => {}
        }
        match Muxer::start(&dir, frame.codec, frame.fps).await {
            Ok(mut started) => {
                if let Err(error) = started.write(&frame.data).await {
                    log::warn!("Desktop recording stopped: {error}");
                    started.finish().await;
                    disabled.store(true, Ordering::Relaxed);
                    break;
                }
                muxer = Some(started);
            }
            Err(error) => {
                log::warn!("Desktop recording unavailable: {error}");
                disabled.store(true, Ordering::Relaxed);
                break;
            }
        }
    }
    if let Some(muxer) = muxer.take() { muxer.finish().await; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_dir_is_off_unless_set_to_a_nonempty_path() {
        assert_eq!(record_dir(None), None);
        assert_eq!(record_dir(Some("")), None);
        assert_eq!(record_dir(Some("   ")), None);
        assert_eq!(record_dir(Some(" /tmp/rec ")), Some(PathBuf::from("/tmp/rec")));
        assert_eq!(record_dir(Some("recordings")), Some(PathBuf::from("recordings")));
    }

    #[test]
    fn utc_timestamps_are_formatted_without_a_date_crate() {
        assert_eq!(timestamp_utc(0), "19700101-000000");
        assert_eq!(timestamp_utc(1_700_000_000), "20231114-221320");
        assert_eq!(timestamp_utc(1_735_689_600), "20250101-000000"); // 2025-01-01T00:00:00Z
        // Leap day: 2024-02-29T12:34:56Z.
        assert_eq!(timestamp_utc(1_709_210_096), "20240229-123456");
    }

    #[test]
    fn segment_paths_are_unique_and_carry_codec_and_stamp() {
        let dir = std::env::temp_dir().join(format!("uu-record-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let first = segment_path(&dir, VideoCodec::H264, "20261022-122500");
        assert_eq!(first.file_name().unwrap().to_string_lossy(), "uu-remote-20261022-122500-h264.mp4");
        std::fs::write(&first, b"x").unwrap();
        let second = segment_path(&dir, VideoCodec::H264, "20261022-122500");
        assert_eq!(second.file_name().unwrap().to_string_lossy(), "uu-remote-20261022-122500-h264-2.mp4");
        let hevc = segment_path(&dir, VideoCodec::H265, "20261022-122500");
        assert_eq!(hevc.file_name().unwrap().to_string_lossy(), "uu-remote-20261022-122500-h265.mp4");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn muxer_args_copy_the_stream_and_tag_hevc_as_hvc1() {
        let h264 = muxer_args(VideoCodec::H264, 60, Path::new("/rec/a.mp4"));
        assert!(h264.windows(2).any(|w| w == ["-f", "h264"]));
        assert!(h264.windows(2).any(|w| w == ["-framerate", "60"]));
        assert!(h264.windows(2).any(|w| w == ["-c:v", "copy"]));
        assert!(h264.windows(2).any(|w| w == ["-f", "mp4"]));
        assert!(!h264.iter().any(|a| a == "hvc1"));
        assert_eq!(h264.last().map(String::as_str), Some("/rec/a.mp4"));

        let h265 = muxer_args(VideoCodec::H265, 30, Path::new("/rec/b.mp4"));
        assert!(h265.windows(2).any(|w| w == ["-f", "hevc"]));
        assert!(h265.windows(2).any(|w| w == ["-tag:v", "hvc1"]));
        assert!(h265.windows(2).any(|w| w == ["-c:v", "copy"]));
    }

    /// End-to-end at the module boundary: synthetic IDR + P access units in, one valid MP4
    /// out — the same shape `stream_desktop` feeds the recorder. Needs a system FFmpeg with
    /// libx264 (already a runtime requirement of the host), so it is ignored by default like
    /// the HEVC capability probe in video.rs.
    #[tokio::test]
    #[ignore = "requires installed FFmpeg with libx264"]
    async fn recorder_writes_a_playable_mp4() {
        // One second of 10 fps synthetic video as separate access units (AUD-delimited).
        let mut command = crate::video::ffmpeg_command();
        command.args(["-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
            "-i", "testsrc=size=64x64:rate=10:duration=1", "-c:v", "libx264", "-profile:v", "baseline",
            "-pix_fmt", "yuv420p", "-g", "5", "-bf", "0",
            "-x264-params", "aud=1:repeat-headers=1:scenecut=0:keyint=5:min-keyint=5", "-f", "h264", "pipe:1"])
            .stdout(Stdio::piped()).stderr(Stdio::null());
        let output = command.output().await.unwrap();
        assert!(output.status.success());
        // Split on AUDs (H.264 NAL type 9) the way the production reader does, and treat a
        // unit holding an IDR slice (type 5) with no non-IDR slice (type 1) as independent —
        // the H.264 rules of video::is_independent_frame_for, restated locally for the test.
        fn nal_types(unit: &[u8]) -> Vec<u8> {
            let mut types = Vec::new();
            let mut i = 0;
            while i + 3 < unit.len() {
                let prefix = if unit[i..].starts_with(&[0, 0, 0, 1]) { 4 }
                    else if unit[i..].starts_with(&[0, 0, 1]) { 3 } else { i += 1; continue };
                if i + prefix < unit.len() { types.push(unit[i + prefix] & 31); }
                i += prefix;
            }
            types
        }
        let mut units: Vec<Vec<u8>> = Vec::new();
        let mut pending = output.stdout.as_slice();
        while !pending.is_empty() {
            let auds: Vec<usize> = {
                let mut found = Vec::new();
                let mut i = 0;
                while i + 3 < pending.len() {
                    let prefix = if pending[i..].starts_with(&[0, 0, 0, 1]) { 4 }
                        else if pending[i..].starts_with(&[0, 0, 1]) { 3 } else { i += 1; continue };
                    if i + prefix < pending.len() && pending[i + prefix] & 31 == 9 { found.push(i); }
                    i += prefix;
                }
                found
            };
            let end = if auds.len() > 1 { auds[1] } else { pending.len() };
            units.push(pending[..end].to_vec());
            pending = &pending[end..];
        }
        assert!(units.len() >= 10);

        let dir = std::env::temp_dir().join(format!("uu-record-e2e-{}", std::process::id()));
        let (tx, rx) = mpsc::channel(QUEUE_FRAMES);
        let disabled = Arc::new(AtomicBool::new(false));
        let task = tokio::spawn(run(dir.clone(), rx, disabled));
        for (index, unit) in units.iter().enumerate() {
            let types = nal_types(unit);
            let independent = types.contains(&5) && !types.contains(&1);
            if index == 0 { assert!(independent, "first access unit must be an IDR"); }
            tx.send(RecordFrame { codec: VideoCodec::H264, fps: 10, independent, data: unit.clone() }).await.unwrap();
        }
        drop(tx); // Session end: the muxer must finalize the file on its own.
        tokio::time::timeout(Duration::from_secs(15), task).await.unwrap().unwrap();

        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "mp4")).collect();
        assert_eq!(files.len(), 1);
        let path = files[0].path();
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
        // ffprobe ships with FFmpeg; a finalized MP4 reports its video stream and duration.
        let probe = tokio::process::Command::new("ffprobe")
            .args(["-hide_banner", "-show_entries", "format=duration", "-of", "csv=p=0", path.to_str().unwrap()])
            .output().await.unwrap();
        assert!(probe.status.success());
        let duration: f64 = String::from_utf8_lossy(&probe.stdout).trim().parse().unwrap();
        assert!((0.5..=1.5).contains(&duration), "duration={duration}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
