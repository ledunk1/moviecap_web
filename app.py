from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx, CompositeVideoClip
import os
import json
import numpy as np
import random
import glob
import subprocess
import platform
import threading
import webbrowser
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class GPUDetector:
    def __init__(self):
        self.gpu_info = {
            'intel': False,
            'amd': False,
            'nvidia': False,
            'opencl': False,
            'encoders': [],
            'decoders': []
        }
        self.detect_gpu()
    
    def detect_gpu(self):
        """Detect available GPU hardware and capabilities"""
        try:
            # Detect GPU hardware
            self._detect_hardware()
            
            # Detect OpenCV OpenCL support
            self._detect_opencl()
            
            # Detect FFmpeg hardware encoders/decoders
            self._detect_ffmpeg_hardware()
            
            self._print_gpu_status()
            
        except Exception as e:
            print(f"Error detecting GPU: {e}")
    
    def _detect_hardware(self):
        """Detect GPU hardware"""
        try:
            if platform.system() == "Windows":
                # Windows GPU detection
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=10)
                gpu_info = result.stdout.lower()
                
                if 'intel' in gpu_info:
                    self.gpu_info['intel'] = True
                if 'amd' in gpu_info or 'radeon' in gpu_info:
                    self.gpu_info['amd'] = True
                if 'nvidia' in gpu_info or 'geforce' in gpu_info or 'quadro' in gpu_info:
                    self.gpu_info['nvidia'] = True
                    
            elif platform.system() == "Linux":
                # Linux GPU detection
                try:
                    result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
                    gpu_info = result.stdout.lower()
                    
                    if 'intel' in gpu_info and 'vga' in gpu_info:
                        self.gpu_info['intel'] = True
                    if 'amd' in gpu_info or 'radeon' in gpu_info:
                        self.gpu_info['amd'] = True
                    if 'nvidia' in gpu_info:
                        self.gpu_info['nvidia'] = True
                except:
                    pass
                    
        except Exception as e:
            print(f"Hardware detection error: {e}")
    
    def _detect_opencl(self):
        """Detect OpenCV OpenCL support"""
        try:
            import cv2
            if cv2.ocl.haveOpenCL():
                self.gpu_info['opencl'] = True
                cv2.ocl.setUseOpenCL(True)
        except ImportError:
            pass
        except Exception as e:
            print(f"OpenCL detection error: {e}")
    
    def _detect_ffmpeg_hardware(self):
        """Detect FFmpeg hardware encoders and decoders"""
        try:
            # Check for hardware encoders
            encoder_result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                          capture_output=True, text=True, timeout=10)
            encoder_output = encoder_result.stdout.lower()
            
            # Check for hardware decoders
            decoder_result = subprocess.run(['ffmpeg', '-hide_banner', '-decoders'], 
                                          capture_output=True, text=True, timeout=10)
            decoder_output = decoder_result.stdout.lower()
            
            # Intel Quick Sync
            if 'h264_qsv' in encoder_output:
                self.gpu_info['encoders'].append('h264_qsv')
            if 'hevc_qsv' in encoder_output:
                self.gpu_info['encoders'].append('hevc_qsv')
            if 'h264_qsv' in decoder_output:
                self.gpu_info['decoders'].append('h264_qsv')
            if 'hevc_qsv' in decoder_output:
                self.gpu_info['decoders'].append('hevc_qsv')
            
            # AMD AMF
            if 'h264_amf' in encoder_output:
                self.gpu_info['encoders'].append('h264_amf')
            if 'hevc_amf' in encoder_output:
                self.gpu_info['encoders'].append('hevc_amf')
            
            # NVIDIA NVENC
            if 'h264_nvenc' in encoder_output:
                self.gpu_info['encoders'].append('h264_nvenc')
            if 'hevc_nvenc' in encoder_output:
                self.gpu_info['encoders'].append('hevc_nvenc')
            
            # Always include software encoder as fallback
            if 'libx264' in encoder_output:
                self.gpu_info['encoders'].append('libx264')
                
        except Exception as e:
            print(f"FFmpeg detection error: {e}")
            # Fallback to software encoding
            self.gpu_info['encoders'] = ['libx264']
    
    def _print_gpu_status(self):
        """Print GPU detection status"""
        print("\n" + "="*50)
        print("🎮 GPU DETECTION RESULTS")
        print("="*50)
        
        if self.gpu_info['intel']:
            print("🔷 Intel GPU detected")
        if self.gpu_info['amd']:
            print("🔴 AMD GPU detected")
        if self.gpu_info['nvidia']:
            print("🟢 NVIDIA GPU detected")
        
        if any([self.gpu_info['intel'], self.gpu_info['amd'], self.gpu_info['nvidia']]):
            print("✅ GPU acceleration available")
        else:
            print("⚠️  No GPU detected - using CPU only")
        
        if self.gpu_info['opencl']:
            print("🔷 OpenCV OpenCL support detected")
            print("✅ OpenCL enabled for OpenCV")
        else:
            print("❌ OpenCV OpenCL not available")
        
        if self.gpu_info['encoders']:
            print(f"🎬 Supported hardware encoders: {', '.join(self.gpu_info['encoders'])}")
        
        if self.gpu_info['decoders']:
            print(f"🎞️ Supported hardware decoders: {', '.join(self.gpu_info['decoders'])}")
        
        print("="*50 + "\n")
    
    def get_best_encoder(self):
        """Get the best available encoder"""
        if 'h264_nvenc' in self.gpu_info['encoders']:
            return 'h264_nvenc'
        elif 'h264_qsv' in self.gpu_info['encoders']:
            return 'h264_qsv'
        elif 'h264_amf' in self.gpu_info['encoders']:
            return 'h264_amf'
        else:
            return 'libx264'

class VideoEditorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Auto Movie Recap Editor")
        self.root.geometry("400x200")
        self.root.resizable(False, False)
        
        # Set window icon if available
        try:
            self.root.iconbitmap('static/movie.ico')
        except:
            pass
        
        self.setup_ui()
        self.server_thread = None
        self.server_running = False
    
    def setup_ui(self):
        """Setup the Tkinter UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Auto Movie Recap Editor", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Server Status: Stopped", 
                                     foreground="red")
        self.status_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        # Start Server button
        self.start_button = ttk.Button(buttons_frame, text="Start Server", 
                                      command=self.start_server, width=15)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        # Open Browser button
        self.browser_button = ttk.Button(buttons_frame, text="Open Browser", 
                                        command=self.open_browser, width=15, 
                                        state="disabled")
        self.browser_button.grid(row=0, column=1, padx=(10, 0))
        
        # Stop Server button
        self.stop_button = ttk.Button(buttons_frame, text="Stop Server", 
                                     command=self.stop_server, width=15, 
                                     state="disabled")
        self.stop_button.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        # Credits
        credits_label = ttk.Label(main_frame, text="Created by Trialota", 
                                 font=('Arial', 10, 'italic'), foreground="gray")
        credits_label.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def start_server(self):
        """Start the Flask server in a separate thread"""
        if not self.server_running:
            self.server_thread = threading.Thread(target=self.run_server, daemon=True)
            self.server_thread.start()
            
            self.server_running = True
            self.status_label.config(text="Server Status: Running", foreground="green")
            self.start_button.config(state="disabled")
            self.browser_button.config(state="normal")
            self.stop_button.config(state="normal")
    
    def run_server(self):
        """Run the Flask server"""
        try:
            app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
        except Exception as e:
            messagebox.showerror("Server Error", f"Failed to start server: {e}")
            self.server_running = False
            self.status_label.config(text="Server Status: Error", foreground="red")
    
    def open_browser(self):
        """Open the web browser to the application"""
        webbrowser.open('http://127.0.0.1:5000')
    
    def stop_server(self):
        """Stop the server (note: this is a simplified implementation)"""
        self.server_running = False
        self.status_label.config(text="Server Status: Stopped", foreground="red")
        self.start_button.config(state="normal")
        self.browser_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        messagebox.showinfo("Server", "Server stop requested. Please close the application to fully stop the server.")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

# Initialize GPU detector
gpu_detector = GPUDetector()

def allowed_file(filename):
    allowed_extensions = {'mp4', 'avi', 'mov', 'mp3', 'wav', 'aac', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_unique_filename(original_filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name, ext = os.path.splitext(secure_filename(original_filename))
    return f"{name}_{timestamp}{ext}"

def get_all_files():
    files = []
    for filepath in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
        if os.path.isfile(filepath):
            filename = os.path.basename(filepath)
            size = os.path.getsize(filepath)
            modified = os.path.getmtime(filepath)
            files.append({
                'name': filename,
                'size': size,
                'modified': datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')
            })
    return files

def get_end_segment(duration):
    """Calculate end segment based on video duration, ensuring last 10 seconds are included"""
    minutes = duration / 60
    if minutes <= 90:
        main_start = duration - 300  # 5 minutes before end
    elif minutes <= 120:
        main_start = duration - 480  # 8 minutes before end
    else:
        main_start = duration - 600  # 10 minutes before end
    
    final_segment_start = duration - 10
    final_segment_end = duration - 0.5
    return (main_start, final_segment_start, final_segment_end)

def analyze_video_quality(video):
    """Analyze input video to determine quality settings"""
    height = video.h
    width = video.w
    pixels = width * height

    if pixels >= 8294400:  # 4K
        return '40000k'
    elif pixels >= 2073600:  # 1080p
        return '10000k'
    elif pixels >= 921600:  # 720p
        return '6000k'
    else:
        return '4000k'

def validate_settings(settings):
    """Validate probability settings total 100%"""
    effect_total = sum(settings['effect_probs'].values())
    if not np.isclose(effect_total, 1.0):
        raise ValueError("Effect probabilities must total 100%")

    segment_total = sum(settings['segment_dist'].values())
    if not np.isclose(segment_total, 1.0):
        raise ValueError("Segment distribution must total 100%")

def create_zoom_effect(clip, zoom_factor=1.3, duration=None):
    if duration is None:
        duration = clip.duration

    def zoom(t):
        progress = np.sin((t/3) * np.pi / duration) * (zoom_factor - 1) + 1
        return progress

    return clip.resize(zoom).crop(
        x_center=clip.w/2,
        y_center=clip.h/2,
        width=clip.w,
        height=clip.h
    )

def create_freeze_frame_with_effects(clip, duration, settings):
    freeze_frame = clip.to_ImageClip(t=0)
    
    if random.random() < settings['freeze']['zoom_probability']:
        effect = create_zoom_effect(freeze_frame, zoom_factor=1.3, duration=duration)
    else:
        effect = freeze_frame
    
    return effect.set_duration(duration)

def apply_fade_transition(clip, fade_duration):
    return clip.fadein(fade_duration).fadeout(fade_duration)

def is_timestamp_excluded(time, excluded_timestamps):
    """Check if a timestamp falls within excluded ranges"""
    return any(start <= time <= end for start, end, _ in excluded_timestamps)

def create_random_effect_clip(clip, start_time, duration, settings):
    if any(is_timestamp_excluded(t, settings['excluded_timestamps'])
           for t in np.arange(start_time, start_time + duration, 0.5)):
        return None

    subclip = clip.subclip(start_time, start_time + duration)
    
    effect_choices = ['slowmo', 'freeze', 'normal']
    weights = [settings['effect_probs']['slowmo'],
              settings['effect_probs']['freeze'],
              settings['effect_probs']['normal']]
    
    effect_type = random.choices(effect_choices, weights=weights)[0]
    
    if effect_type == "slowmo":
        effect_clip = subclip.fx(vfx.speedx, settings['slowmo_speed'])
    elif effect_type == "freeze":
        effect_clip = create_freeze_frame_with_effects(subclip, duration, settings)
    else:  # normal speed
        effect_clip = subclip
    
    if random.random() < settings['transition']['fade_probability']:
        effect_clip = apply_fade_transition(effect_clip, settings['transition']['fade_duration'])
    
    return effect_clip

def create_section_clips(video, section_start, section_end, num_segments, settings):
    clips = []
    section_duration = section_end - section_start
    
    actual_segments = max(1, min(num_segments, int(section_duration / 2)))
    step = section_duration / actual_segments
    
    current_time = section_start
    while current_time < section_end:
        remaining_time = section_end - current_time
        if remaining_time < 2:
            break
            
        max_duration = min(4, remaining_time)
        segment_duration = random.uniform(2, max_duration)
        
        try:
            clip = create_random_effect_clip(video, current_time, segment_duration, settings)
            if clip is not None:
                clips.append(clip)
                
                if random.random() < settings['repeat']['probability']:
                    num_repeats = random.randint(1, settings['repeat']['max_repeats'])
                    for _ in range(num_repeats):
                        repeat_clip = create_random_effect_clip(video, current_time, segment_duration, settings)
                        if repeat_clip is not None:
                            clips.append(repeat_clip)
        except Exception as e:
            print(f"Error creating clip at {current_time}: {e}")
            
        step_variation = random.uniform(0.8, 1.2)
        current_time += step * step_variation
        
    return clips

def create_included_segment_clips(video, include_timestamps, settings):
    """Create clips from specifically included timestamps"""
    included_clips = []

    for start, end, position in include_timestamps:
        if position not in ['beginning', 'middle', 'end']:
            position = 'end'  # Default to end if invalid position

        try:
            duration = end - start
            if duration <= 0:
                continue

            subclip = video.subclip(start, end)

            # Apply random effects to included clips
            effect_choices = ['slowmo', 'freeze', 'normal']
            weights = [settings['effect_probs']['slowmo'],
                      settings['effect_probs']['freeze'],
                      settings['effect_probs']['normal']]

            effect_type = random.choices(effect_choices, weights=weights)[0]

            if effect_type == "slowmo":
                effect_clip = subclip.fx(vfx.speedx, settings['slowmo_speed'])
            elif effect_type == "freeze":
                effect_clip = create_freeze_frame_with_effects(subclip, duration, settings)
            else:  # normal speed
                effect_clip = subclip

            if random.random() < settings['transition']['fade_probability']:
                effect_clip = apply_fade_transition(effect_clip, settings['transition']['fade_duration'])

            included_clips.append((effect_clip, position))

        except Exception as e:
            print(f"Error creating included clip at {start}-{end}: {e}")

    return included_clips

@app.route('/')
def index():
    try:
        files = get_all_files()
        return render_template('index.html', files=files)
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('index.html', files=[])

@app.route('/files', methods=['GET'])
def list_files():
    files = get_all_files()
    return jsonify(files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    video_file = request.files['video']
    audio_file = request.files['audio']
    settings = json.loads(request.form.get('settings', '{}'))
    
    if video_file.filename == '' or audio_file.filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    if not (allowed_file(video_file.filename) and allowed_file(audio_file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Generate unique filenames
        video_filename = generate_unique_filename(video_file.filename)
        audio_filename = generate_unique_filename(audio_file.filename)
        output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        video_file.save(video_path)
        audio_file.save(audio_path)
        
        # Load video and audio
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        # Update settings with GPU-optimized encoder
        best_encoder = gpu_detector.get_best_encoder()
        settings['quality'] = {
            'bitrate': analyze_video_quality(video),
            'encoder': best_encoder
        }
        
        print(f"🎬 Using encoder: {best_encoder}")
        
        video_duration = video.duration
        target_duration = audio.duration
        
        # Calculate video sections
        beginning_end = video_duration * 0.33
        middle_start = video_duration * 0.33
        middle_end = video_duration * 0.66
        main_start, final_start, final_end = get_end_segment(video_duration)
        
        # Calculate segments
        beginning_target = target_duration * settings['segment_dist']['beginning']
        middle_target = target_duration * settings['segment_dist']['middle']
        end_target = target_duration * settings['segment_dist']['end']
        
        avg_segment_duration = 3
        beginning_segments = int(beginning_target / avg_segment_duration)
        middle_segments = int(middle_target / avg_segment_duration)
        end_segments = int(end_target / avg_segment_duration)
        
        # Create clips
        beginning_clips = create_section_clips(video, 0, beginning_end, beginning_segments, settings)
        middle_clips = create_section_clips(video, middle_start, middle_end, middle_segments, settings)
        end_clips = create_section_clips(video, main_start, final_start, end_segments, settings)
        
        # Add final clip
        final_clip = video.subclip(final_start, final_end)
        end_clips.append(final_clip)
        
        # Create included segment clips
        included_clips = create_included_segment_clips(video, settings['include_timestamps'], settings)
        
        # Organize all segments
        all_segments = beginning_clips + middle_clips + end_clips
        
        # Add specifically included clips at their specified positions
        included_end_clips = [clip for clip, position in included_clips if position == 'end']
        
        # Calculate total duration
        main_clips_duration = sum(clip.duration for clip in all_segments)
        included_clips_duration = sum(clip.duration for clip, _ in included_clips)
        
        # Trim if necessary
        if main_clips_duration + included_clips_duration > target_duration:
            trim_duration = main_clips_duration + included_clips_duration - target_duration
            while trim_duration > 0 and all_segments:
                clip_to_remove = all_segments.pop()
                trim_duration -= clip_to_remove.duration
        
        # Add included clips
        all_segments.extend(included_end_clips)
        
        if not all_segments:
            raise ValueError("No valid video segments could be created")
        
        final_video = concatenate_videoclips(all_segments)
        final_video.fps = settings['fps']
        
        if final_video.duration > target_duration:
            final_video = final_video.subclip(0, target_duration)
        
        final_video = final_video.set_audio(audio)
        
        # Write final video with GPU acceleration if available
        codec = settings['quality']['encoder']
        preset = 'slow' if codec == 'libx264' else 'medium'
        
        print(f"🎞️ Rendering with codec: {codec}, preset: {preset}")
        
        final_video.write_videofile(
            output_path,
            codec=codec,
            audio_codec='aac',
            bitrate=settings['quality']['bitrate'],
            preset=preset,
            threads=4,
            fps=settings['fps']
        )
        
        # Cleanup
        video.close()
        audio.close()
        final_video.close()
        
        # Clean up input files after processing
        os.remove(video_path)
        os.remove(audio_path)
        
        return jsonify({
            'output': output_filename,
            'message': 'Processing completed successfully',
            'encoder_used': codec
        })
        
    except Exception as e:
        # Clean up any files in case of error
        for path in [video_path, audio_path, output_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'message': 'File deleted successfully'})
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check if running in GUI mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        # Run with Tkinter GUI
        gui = VideoEditorGUI()
        gui.run()
    else:
        # Run normally
        app.run(host='0.0.0.0', port=5000, debug=True)