from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx, CompositeVideoClip
import numpy as np
import random

# File paths
movie_path = '/content/drive/MyDrive/EDIT/1/Better.Days.2019.CHINESE.1080p.BluRay.H264.AAC-VXT.mp4'
audio_path = '/content/drive/MyDrive/EDIT/1/combined.mp3'

def get_end_segment(duration):
    """Calculate end segment based on video duration, ensuring last 10 seconds are included"""
    # Convert duration to minutes for easier calculation
    minutes = duration / 60

    # Calculate the main segment start time
    if minutes <= 90:
        main_start = duration - 300  # 5 minutes before end
    elif minutes <= 120:
        main_start = duration - 480  # 8 minutes before end
    else:
        main_start = duration - 600  # 10 minutes before end

    # Force inclusion of last 10 seconds
    final_segment_start = duration - 10
    final_segment_end = duration - 0.5  # Stop half second before end to avoid abrupt cuts

    return (main_start, final_segment_start, final_segment_end)

# Effect settings
settings = {
    'effect_probs': {
        'slowmo': 0.25,
        'freeze': 0.65,
        'normal': 0.10
    },
    'segment_dist': {
        'beginning': 0.40,  # 40%
        'middle': 0.25,     # 35%
        'end': 0.35         # 25%
    },
    'slowmo_speed': 0.35,
    'transition': {
        'fade_probability': 0.5,
        'fade_duration': 0.5
    },
    'repeat': {
        'probability': 0.2,  # 20% chance to repeat a segment
        'max_repeats': 1     # Maximum number of times a segment can be repeated
    },
    'freeze': {
        'zoom_probability': 0.3  # 30% chance to add zoom effect to freeze frames
    },
    'quality': {
        'maintain_quality': True,  # Preserve input video quality
        'bitrate': None,  # Will be set based on input video
        'preset': 'slow'  # Higher quality encoding
    },
    'fps': 24,
    'excluded_timestamps': [
        (0, 97),      # Opening sequence
        (176, 206),   # Scene transition
        (260, 280),   # Scene transition
        (342, 357),   # Scene transition
        (411, 442),   # Scene transition
        (7522, 7642), # End sequence
        (7656, 7668), # Credits start
        (7773, 8138)  # End credits
    ],
    # NEW FEATURE: Include specific timestamps that should be forced into the final recap
    # Format: (timestamp_start, timestamp_end, position)
    # Position: 'end' means this clip will be placed at the end of the final recap
    'include_timestamps': [
        (7674, 7773, 'end')  # Important scene to include at the end
    ]
}

def analyze_video_quality(video):
    """Analyze input video to determine quality settings"""
    height = video.h
    width = video.w
    pixels = width * height

    if pixels >= 8294400:  # 4K (3840x2160)
        return '40000k'
    elif pixels >= 2073600:  # 1080p (1920x1080)
        return '10000k'
    elif pixels >= 921600:  # 720p (1280x720)
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

def is_timestamp_excluded(time, excluded_timestamps):
    """Check if a timestamp falls within excluded ranges"""
    return any(start <= time <= end for start, end in excluded_timestamps)

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
            clip = create_random_effect_clip(
                video,
                current_time,
                segment_duration,
                settings
            )
            if clip is not None:
                clips.append(clip)

                if random.random() < settings['repeat']['probability']:
                    num_repeats = random.randint(1, settings['repeat']['max_repeats'])
                    for _ in range(num_repeats):
                        repeat_clip = create_random_effect_clip(
                            video,
                            current_time,
                            segment_duration,
                            settings
                        )
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

try:
    print("Loading video and audio...")
    video = VideoFileClip(movie_path, audio=True)
    audio = AudioFileClip(audio_path)

    settings['quality']['bitrate'] = analyze_video_quality(video)
    validate_settings(settings)

    video_duration = video.duration
    target_duration = audio.duration

    # Calculate video sections
    beginning_end = video_duration * 0.33
    middle_start = video_duration * 0.33
    middle_end = video_duration * 0.66

    # Get end segment timing including the last 10 seconds
    main_start, final_start, final_end = get_end_segment(video_duration)

    # Calculate target durations for each section
    beginning_target = target_duration * settings['segment_dist']['beginning']
    middle_target = target_duration * settings['segment_dist']['middle']
    end_target = target_duration * settings['segment_dist']['end']

    avg_segment_duration = 3
    beginning_segments = int(beginning_target / avg_segment_duration)
    middle_segments = int(middle_target / avg_segment_duration)
    end_segments = int(end_target / avg_segment_duration)

    print("Creating video segments...")
    beginning_clips = create_section_clips(video, 0, beginning_end, beginning_segments, settings)
    middle_clips = create_section_clips(video, middle_start, middle_end, middle_segments, settings)

    # Create main end section clips
    end_clips = create_section_clips(video, main_start, final_start, end_segments, settings)

    # Add the final 10 seconds clip
    final_clip = video.subclip(final_start, final_end)
    end_clips.append(final_clip)

    # Create included segment clips
    print("Processing specifically included segments...")
    included_clips = create_included_segment_clips(video, settings['include_timestamps'], settings)

    # Organize all segments
    all_segments = beginning_clips + middle_clips + end_clips

    # Add specifically included clips at the end (or other specified positions if implemented)
    included_end_clips = [clip for clip, position in included_clips if position == 'end']

    # Calculate how much time we have left after including required clips
    main_clips_duration = sum(clip.duration for clip in all_segments)
    included_clips_duration = sum(clip.duration for clip, _ in included_clips)

    # If we exceed the target duration with included clips, we need to trim the main segments
    if main_clips_duration + included_clips_duration > target_duration:
        trim_duration = main_clips_duration + included_clips_duration - target_duration
        # Remove clips from the end of the main segments until we have enough time
        while trim_duration > 0 and all_segments:
            clip_to_remove = all_segments.pop()
            trim_duration -= clip_to_remove.duration

    # Add the included end clips to the end of our segments
    all_segments.extend(included_end_clips)

    if not all_segments:
        raise ValueError("No valid video segments could be created")

    print("Concatenating clips...")
    final_video = concatenate_videoclips(all_segments)
    final_video.fps = settings['fps']

    if final_video.duration > target_duration:
        final_video = final_video.subclip(0, target_duration)

    final_video = final_video.set_audio(audio)

    output_path = '/content/drive/MyDrive/EDIT/1/fix_11.mp4'
    print("Rendering final video...")
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        bitrate=settings['quality']['bitrate'],
        preset=settings['quality']['preset'],
        threads=4,
        fps=settings['fps']
    )

    print(f"Video saved to: {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    try:
        video.close()
        audio.close()
        final_video.close()
    except:
        pass