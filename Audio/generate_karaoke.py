import pygame
import json
import soundfile as sf
import time
import sys


def generate_karaoke(audio_path, segments):
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Karaoke")

    # Load audio
    audio_data, samplerate = sf.read(audio_path)
    pygame.mixer.init(frequency=samplerate)
    sound = pygame.mixer.Sound(audio_path)

    # Define font
    font = pygame.font.Font(None, 74)

    running = True
    start_time = time.time()

    # Play audio
    sound.play()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        current_time = time.time() - start_time
        current_segment = None

        for segment in segments:
            if segment["start"] <= current_time <= segment["end"]:
                current_segment = segment
                break

        if current_segment:
            text = current_segment["text"]
            text_surf = font.render(text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(400, 300))
            screen.blit(text_surf, text_rect)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    audio_path = sys.argv[1]
    with open(sys.argv[2], "r") as f:
        segments = json.load(f)["segments"]

    generate_karaoke(audio_path, segments)
