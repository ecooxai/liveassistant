use crate::backends::PlaybackCommand;
use crate::AecError;
use flume::{Receiver, Sender};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{class, msg_send};
use objc2_foundation::{NSError, NSString};
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;

// ============================================================================
// AudioToolbox FFI Types and Constants
// ============================================================================

type OSStatus = i32;
type AudioComponentInstance = *mut c_void;
type AudioComponent = *mut c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AudioComponentDescription {
    component_type: u32,
    component_sub_type: u32,
    component_manufacturer: u32,
    component_flags: u32,
    component_flags_mask: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioBufferList {
    number_buffers: u32,
    buffers: [AudioBuffer; 1], // Variable length, but we use 1 for mono
}

#[repr(C)]
struct AudioBuffer {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut c_void,
}

#[repr(C)]
struct AudioTimeStamp {
    sample_time: f64,
    host_time: u64,
    rate_scalar: f64,
    word_clock_time: u64,
    smpte_time: [u8; 24], // SMPTETime struct
    flags: u32,
    reserved: u32,
}

type AURenderCallback = extern "C" fn(
    in_ref_con: *mut c_void,
    io_action_flags: *mut u32,
    in_time_stamp: *const AudioTimeStamp,
    in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut AudioBufferList,
) -> OSStatus;

#[repr(C)]
struct AURenderCallbackStruct {
    input_proc: AURenderCallback,
    input_proc_ref_con: *mut c_void,
}

// Audio Unit Types
const K_AUDIO_UNIT_TYPE_OUTPUT: u32 = 0x61756f75; // 'auou'
const K_AUDIO_UNIT_SUB_TYPE_VOICE_PROCESSING_IO: u32 = 0x7670696f; // 'vpio'
const K_AUDIO_UNIT_MANUFACTURER_APPLE: u32 = 0x6170706c; // 'appl'

// Audio Unit Properties
const K_AUDIO_OUTPUT_UNIT_PROPERTY_ENABLE_IO: u32 = 2003;
const K_AUDIO_OUTPUT_UNIT_PROPERTY_SET_INPUT_CALLBACK: u32 = 2005;
const K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT: u32 = 8;
const K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: u32 = 23;
const K_AU_VOICE_IO_PROPERTY_BYPASS_VOICE_PROCESSING: u32 = 2100;

// Audio Unit Scopes
const K_AUDIO_UNIT_SCOPE_INPUT: u32 = 1;
const K_AUDIO_UNIT_SCOPE_OUTPUT: u32 = 2;
const K_AUDIO_UNIT_SCOPE_GLOBAL: u32 = 0;

// Audio Format
const K_AUDIO_FORMAT_LINEAR_PCM: u32 = 0x6c70636d; // 'lpcm'
const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1 << 0;
const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 1 << 3;
const K_AUDIO_FORMAT_FLAG_IS_NON_INTERLEAVED: u32 = 1 << 5;

// Audio Session
const AV_AUDIO_SESSION_CATEGORY_PLAY_AND_RECORD: &str = "AVAudioSessionCategoryPlayAndRecord";
const AV_AUDIO_SESSION_MODE_VIDEO_CHAT: &str = "AVAudioSessionModeVideoChat";

// Audio session options bitmask:
// - 0x1 = AVAudioSessionCategoryOptionDefaultToSpeaker
// - 0x4 = AVAudioSessionCategoryOptionAllowBluetooth
// - 0x20 = AVAudioSessionCategoryOptionAllowBluetoothA2DP
const AV_AUDIO_SESSION_OPTIONS: u64 = 0x1 | 0x4 | 0x20;

const BUFFER_SIZE: u32 = 1024;
const SAMPLE_RATE: f64 = 48000.0;

// ============================================================================
// FFI Declarations
// ============================================================================

extern "C" {
    fn AudioComponentFindNext(
        component: AudioComponent,
        desc: *const AudioComponentDescription,
    ) -> AudioComponent;

    fn AudioComponentInstanceNew(
        component: AudioComponent,
        out_instance: *mut AudioComponentInstance,
    ) -> OSStatus;

    fn AudioComponentInstanceDispose(instance: AudioComponentInstance) -> OSStatus;

    fn AudioUnitSetProperty(
        unit: AudioComponentInstance,
        property_id: u32,
        scope: u32,
        element: u32,
        data: *const c_void,
        data_size: u32,
    ) -> OSStatus;

    fn AudioUnitGetProperty(
        unit: AudioComponentInstance,
        property_id: u32,
        scope: u32,
        element: u32,
        data: *mut c_void,
        data_size: *mut u32,
    ) -> OSStatus;

    fn AudioUnitInitialize(unit: AudioComponentInstance) -> OSStatus;

    fn AudioUnitUninitialize(unit: AudioComponentInstance) -> OSStatus;

    fn AudioOutputUnitStart(unit: AudioComponentInstance) -> OSStatus;

    fn AudioOutputUnitStop(unit: AudioComponentInstance) -> OSStatus;

    fn AudioUnitRender(
        unit: AudioComponentInstance,
        io_action_flags: *mut u32,
        in_time_stamp: *const AudioTimeStamp,
        in_output_bus_number: u32,
        in_number_frames: u32,
        io_data: *mut AudioBufferList,
    ) -> OSStatus;
}

// ============================================================================
// Callback Context
// ============================================================================

const MAX_FRAMES_PER_CALLBACK: usize = 4096;

struct VPIOContext {
    audio_unit: AudioComponentInstance,
    capture_sender: Sender<Vec<f32>>,
    playback_buffer: Arc<Mutex<Vec<f32>>>,
    // Pre-allocated scratch buffer to avoid heap allocation in callback
    input_scratch: std::sync::Mutex<Vec<f32>>,
    sample_rate: f64,
}

unsafe impl Send for VPIOContext {}
unsafe impl Sync for VPIOContext {}

// ============================================================================
// Public API
// ============================================================================

/// Create iOS VPIO (Voice Processing I/O) capture backend.
/// Uses low-level Audio Unit for reliable AEC.
/// Returns (sample_rate, buffer_size).
pub fn create_backend(
    public_sender: Sender<Vec<f32>>,
    playback_rx: Receiver<PlaybackCommand>,
) -> Result<(u32, usize), AecError> {
    // Configure audio session first (on main thread context is fine)
    configure_audio_session()?;

    // Create VPIO unit
    let audio_unit = create_vpio_unit()?;

    // Enable I/O on both buses
    enable_io(audio_unit)?;

    // Set audio format on both buses
    let format = create_audio_format(SAMPLE_RATE, 1); // Mono
    set_audio_format(audio_unit, &format)?;

    // Disable voice processing bypass (ensure AEC is ON)
    let bypass: u32 = 0;
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            K_AU_VOICE_IO_PROPERTY_BYPASS_VOICE_PROCESSING,
            K_AUDIO_UNIT_SCOPE_GLOBAL,
            0,
            &bypass as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u32,
        )
    };
    if status != 0 {
        eprintln!("[sys-voice] Warning: Could not set voice processing bypass: {status}");
    }

    // Create context for callbacks
    let context = Box::new(VPIOContext {
        audio_unit,
        capture_sender: public_sender,
        playback_buffer: Arc::new(Mutex::new(Vec::new())),
        input_scratch: std::sync::Mutex::new(vec![0.0f32; MAX_FRAMES_PER_CALLBACK]),
        sample_rate: SAMPLE_RATE,
    });
    let context_ptr = Box::into_raw(context);

    // Set render callback for output (Bus 0) - provides playback samples for AEC reference
    set_render_callback(audio_unit, render_callback, context_ptr as *mut c_void)?;

    // Set input callback for capture (Bus 1) - receives AEC-processed samples
    set_input_callback(audio_unit, input_callback, context_ptr as *mut c_void)?;

    // Initialize the audio unit
    let status = unsafe { AudioUnitInitialize(audio_unit) };
    if status != 0 {
        unsafe {
            let _ = Box::from_raw(context_ptr);
        }
        return Err(AecError::BackendError(format!(
            "AudioUnitInitialize failed: {status}"
        )));
    }

    // Start the audio unit
    let status = unsafe { AudioOutputUnitStart(audio_unit) };
    if status != 0 {
        unsafe {
            AudioUnitUninitialize(audio_unit);
            let _ = Box::from_raw(context_ptr);
        }
        return Err(AecError::BackendError(format!(
            "AudioOutputUnitStart failed: {status}"
        )));
    }

    eprintln!("[sys-voice] VPIO Audio Unit started successfully");
    eprintln!("[sys-voice] Sample rate: {SAMPLE_RATE} Hz, Buffer size: {BUFFER_SIZE}");

    // Spawn thread to handle playback requests
    let playback_buffer = unsafe { (*context_ptr).playback_buffer.clone() };

    std::thread::Builder::new()
        .name("ios-playback".to_string())
        .spawn(move || {
            run_playback_loop(playback_rx, playback_buffer);
        })
        .map_err(|e| AecError::BackendError(format!("Failed to spawn playback thread: {e}")))?;

    Ok((SAMPLE_RATE as u32, BUFFER_SIZE as usize))
}

// ============================================================================
// Audio Session Configuration
// ============================================================================

fn configure_audio_session() -> Result<(), AecError> {
    let session_class = class!(AVAudioSession);
    let session: Retained<AnyObject> = unsafe { msg_send![session_class, sharedInstance] };

    let category = NSString::from_str(AV_AUDIO_SESSION_CATEGORY_PLAY_AND_RECORD);
    let mode = NSString::from_str(AV_AUDIO_SESSION_MODE_VIDEO_CHAT);

    let mut error: *mut NSError = ptr::null_mut();

    // Set category with DefaultToSpeaker, AllowBluetooth, AllowBluetoothA2DP
    let success: bool = unsafe {
        msg_send![
            &session,
            setCategory: &*category,
            mode: &*mode,
            options: AV_AUDIO_SESSION_OPTIONS,
            error: &mut error
        ]
    };

    if !success {
        let msg = extract_nserror_message(error);
        return Err(AecError::BackendError(format!(
            "Failed to set audio session category: {msg}"
        )));
    }

    // Activate session
    let mut error: *mut NSError = ptr::null_mut();
    let activated: bool = unsafe { msg_send![&session, setActive: true, error: &mut error] };

    if !activated {
        let msg = extract_nserror_message(error);
        return Err(AecError::BackendError(format!(
            "Failed to activate audio session: {msg}"
        )));
    }

    eprintln!("[sys-voice] Audio session configured with VideoChat mode and speaker options");
    Ok(())
}

// ============================================================================
// VPIO Unit Setup
// ============================================================================

fn create_vpio_unit() -> Result<AudioComponentInstance, AecError> {
    let desc = AudioComponentDescription {
        component_type: K_AUDIO_UNIT_TYPE_OUTPUT,
        component_sub_type: K_AUDIO_UNIT_SUB_TYPE_VOICE_PROCESSING_IO,
        component_manufacturer: K_AUDIO_UNIT_MANUFACTURER_APPLE,
        component_flags: 0,
        component_flags_mask: 0,
    };

    let component = unsafe { AudioComponentFindNext(ptr::null_mut(), &desc) };
    if component.is_null() {
        return Err(AecError::BackendError(
            "Could not find VPIO audio component".to_string(),
        ));
    }

    let mut audio_unit: AudioComponentInstance = ptr::null_mut();
    let status = unsafe { AudioComponentInstanceNew(component, &mut audio_unit) };
    if status != 0 {
        return Err(AecError::BackendError(format!(
            "AudioComponentInstanceNew failed: {status}"
        )));
    }

    eprintln!("[sys-voice] Created VPIO Audio Unit");
    Ok(audio_unit)
}

fn enable_io(audio_unit: AudioComponentInstance) -> Result<(), AecError> {
    let enable: u32 = 1;
    let disable: u32 = 0;

    // Enable input on Bus 1 (microphone)
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            K_AUDIO_OUTPUT_UNIT_PROPERTY_ENABLE_IO,
            K_AUDIO_UNIT_SCOPE_INPUT,
            1, // Bus 1 = input
            &enable as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u32,
        )
    };
    if status != 0 {
        return Err(AecError::BackendError(format!(
            "Failed to enable input on Bus 1: {status}"
        )));
    }

    // Enable output on Bus 0 (speaker)
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            K_AUDIO_OUTPUT_UNIT_PROPERTY_ENABLE_IO,
            K_AUDIO_UNIT_SCOPE_OUTPUT,
            0, // Bus 0 = output
            &enable as *const u32 as *const c_void,
            std::mem::size_of::<u32>() as u32,
        )
    };
    if status != 0 {
        return Err(AecError::BackendError(format!(
            "Failed to enable output on Bus 0: {status}"
        )));
    }

    eprintln!("[sys-voice] Enabled I/O on Bus 0 (output) and Bus 1 (input)");
    Ok(())
}

fn create_audio_format(sample_rate: f64, channels: u32) -> AudioStreamBasicDescription {
    AudioStreamBasicDescription {
        sample_rate,
        format_id: K_AUDIO_FORMAT_LINEAR_PCM,
        format_flags: K_AUDIO_FORMAT_FLAG_IS_FLOAT | K_AUDIO_FORMAT_FLAG_IS_PACKED,
        bytes_per_packet: 4 * channels,
        frames_per_packet: 1,
        bytes_per_frame: 4 * channels,
        channels_per_frame: channels,
        bits_per_channel: 32,
        reserved: 0,
    }
}

fn set_audio_format(
    audio_unit: AudioComponentInstance,
    format: &AudioStreamBasicDescription,
) -> Result<(), AecError> {
    // Set format for output scope of Bus 1 (what we receive from mic)
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
            K_AUDIO_UNIT_SCOPE_OUTPUT,
            1, // Bus 1
            format as *const AudioStreamBasicDescription as *const c_void,
            std::mem::size_of::<AudioStreamBasicDescription>() as u32,
        )
    };
    if status != 0 {
        return Err(AecError::BackendError(format!(
            "Failed to set format on Bus 1 output scope: {status}"
        )));
    }

    // Set format for input scope of Bus 0 (what we send to speaker)
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
            K_AUDIO_UNIT_SCOPE_INPUT,
            0, // Bus 0
            format as *const AudioStreamBasicDescription as *const c_void,
            std::mem::size_of::<AudioStreamBasicDescription>() as u32,
        )
    };
    if status != 0 {
        return Err(AecError::BackendError(format!(
            "Failed to set format on Bus 0 input scope: {status}"
        )));
    }

    eprintln!(
        "[sys-voice] Set audio format: {:.0} Hz, {} channels, 32-bit float",
        format.sample_rate, format.channels_per_frame
    );
    Ok(())
}

fn set_render_callback(
    audio_unit: AudioComponentInstance,
    callback: AURenderCallback,
    context: *mut c_void,
) -> Result<(), AecError> {
    let callback_struct = AURenderCallbackStruct {
        input_proc: callback,
        input_proc_ref_con: context,
    };

    // Render callback on Bus 0 (output) - provides samples for speaker/AEC reference
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK,
            K_AUDIO_UNIT_SCOPE_INPUT,
            0, // Bus 0
            &callback_struct as *const AURenderCallbackStruct as *const c_void,
            std::mem::size_of::<AURenderCallbackStruct>() as u32,
        )
    };

    if status != 0 {
        return Err(AecError::BackendError(format!(
            "Failed to set render callback on Bus 0: {status}"
        )));
    }

    eprintln!("[sys-voice] Set render callback on Bus 0 (output)");
    Ok(())
}

fn set_input_callback(
    audio_unit: AudioComponentInstance,
    callback: AURenderCallback,
    context: *mut c_void,
) -> Result<(), AecError> {
    let callback_struct = AURenderCallbackStruct {
        input_proc: callback,
        input_proc_ref_con: context,
    };

    // Input callback on Bus 1 (input) - notified when mic data is ready
    let status = unsafe {
        AudioUnitSetProperty(
            audio_unit,
            K_AUDIO_OUTPUT_UNIT_PROPERTY_SET_INPUT_CALLBACK,
            K_AUDIO_UNIT_SCOPE_GLOBAL,
            0, // Element 0
            &callback_struct as *const AURenderCallbackStruct as *const c_void,
            std::mem::size_of::<AURenderCallbackStruct>() as u32,
        )
    };

    if status != 0 {
        return Err(AecError::BackendError(format!(
            "Failed to set input callback: {status}"
        )));
    }

    eprintln!("[sys-voice] Set input callback on Bus 1 (input)");
    Ok(())
}

// ============================================================================
// Audio Callbacks
// ============================================================================

/// Render callback for Bus 0 (output/speaker)
/// Called when the hardware needs audio samples to play
/// These samples are used as the AEC reference signal
extern "C" fn render_callback(
    in_ref_con: *mut c_void,
    _io_action_flags: *mut u32,
    _in_time_stamp: *const AudioTimeStamp,
    _in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut AudioBufferList,
) -> OSStatus {
    if in_ref_con.is_null() || io_data.is_null() {
        return 0;
    }

    let context = unsafe { &*(in_ref_con as *const VPIOContext) };
    let buffer_list = unsafe { &mut *io_data };

    if buffer_list.number_buffers == 0 {
        return 0;
    }

    let buffer = &mut buffer_list.buffers[0];
    let data = buffer.data as *mut f32;
    let frame_count = in_number_frames as usize;

    if data.is_null() {
        return 0;
    }

    // Try to get samples from playback buffer
    let mut playback_buffer = match context.playback_buffer.try_lock() {
        Ok(b) => b,
        Err(_) => {
            // Can't get lock, output silence
            unsafe {
                ptr::write_bytes(data, 0, frame_count);
            }
            return 0;
        }
    };

    // Copy available samples or pad with silence
    let available = playback_buffer.len().min(frame_count);
    if available > 0 {
        unsafe {
            ptr::copy_nonoverlapping(playback_buffer.as_ptr(), data, available);
        }
        playback_buffer.drain(..available);
    }

    // Fill remaining with silence
    if available < frame_count {
        unsafe {
            // write_bytes already multiplies count by size_of::<T>(), so just pass element count
            ptr::write_bytes(data.add(available), 0, frame_count - available);
        }
    }

    0
}

/// Input callback for Bus 1 (input/microphone)
/// Called when AEC-processed audio is available from the microphone
extern "C" fn input_callback(
    in_ref_con: *mut c_void,
    io_action_flags: *mut u32,
    in_time_stamp: *const AudioTimeStamp,
    in_bus_number: u32,
    in_number_frames: u32,
    _io_data: *mut AudioBufferList,
) -> OSStatus {
    // Static counter for debug logging (only log occasionally to avoid spam)
    static CALLBACK_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let count = CALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Wrap in catch_unwind to prevent panics from crossing FFI boundary
    let result = std::panic::catch_unwind(|| {
        if in_ref_con.is_null() {
            if count < 5 {
                eprintln!("[sys-voice] input_callback: null ref_con");
            }
            return 0;
        }

        let context = unsafe { &*(in_ref_con as *const VPIOContext) };
        let frame_count = (in_number_frames as usize).min(MAX_FRAMES_PER_CALLBACK);

        if count < 5 {
            eprintln!(
                "[sys-voice] input_callback #{}: bus={}, frames={}",
                count, in_bus_number, in_number_frames
            );
        }

        // Use pre-allocated scratch buffer instead of heap allocation
        let mut scratch_guard = match context.input_scratch.try_lock() {
            Ok(g) => g,
            Err(_) => {
                if count < 5 {
                    eprintln!("[sys-voice] input_callback: lock contention");
                }
                return 0; // Can't get lock, skip this callback
            }
        };

        // Ensure scratch buffer is large enough
        if scratch_guard.len() < frame_count {
            scratch_guard.resize(frame_count, 0.0);
        }

        let mut buffer = AudioBuffer {
            number_channels: 1,
            data_byte_size: (frame_count * std::mem::size_of::<f32>()) as u32,
            data: scratch_guard.as_mut_ptr() as *mut c_void,
        };

        let mut buffer_list = AudioBufferList {
            number_buffers: 1,
            buffers: [buffer],
        };

        // Render audio from the input bus (Bus 1)
        let status = unsafe {
            AudioUnitRender(
                context.audio_unit,
                io_action_flags,
                in_time_stamp,
                1, // Bus 1 = input
                in_number_frames,
                &mut buffer_list,
            )
        };

        if status != 0 {
            if count < 10 {
                eprintln!("[sys-voice] AudioUnitRender failed: {}", status);
            }
            return status;
        }

        // Copy to new vec for sending (allocation happens here, outside real-time critical path)
        // Note: This is still an allocation, but it's unavoidable with current channel design
        // A ring buffer would be better for production
        let samples = scratch_guard[..frame_count].to_vec();
        match context.capture_sender.try_send(samples) {
            Ok(_) => {
                if count < 5 {
                    eprintln!("[sys-voice] Sent {} samples", frame_count);
                }
            }
            Err(e) => {
                if count < 10 {
                    eprintln!("[sys-voice] try_send failed: {:?}", e);
                }
            }
        }

        0
    });

    match result {
        Ok(status) => status,
        Err(_) => {
            eprintln!("[sys-voice] input_callback panicked!");
            -1 // Return error to CoreAudio on panic
        }
    }
}

// ============================================================================
// Utilities
// ============================================================================

fn extract_nserror_message(error: *mut NSError) -> String {
    if error.is_null() {
        return "unknown error".to_string();
    }

    unsafe {
        let description: Retained<NSString> = msg_send![error, localizedDescription];
        description.to_string()
    }
}

fn run_playback_loop(
    playback_rx: Receiver<PlaybackCommand>,
    playback_buffer: Arc<Mutex<Vec<f32>>>,
) {
    while let Ok(command) = playback_rx.recv() {
        match command {
            PlaybackCommand::OneShot(samples) => {
                let Ok(mut buffer) = playback_buffer.lock() else {
                    continue;
                };
                buffer.extend(samples);
            }
            PlaybackCommand::StartStream(chunk_rx) => {
                let buf = playback_buffer.clone();
                std::thread::spawn(move || {
                    while let Ok(samples) = chunk_rx.recv() {
                        let Ok(mut b) = buf.lock() else {
                            break;
                        };
                        b.extend(samples);
                    }
                });
            }
        }
    }
}
