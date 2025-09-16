import asyncio

import edge_tts



async def generate_mongolian_tts(text, filename="mongolian-yesui.mp3"):

    # Set the Mongolian neural voice

    voice = "mn-MN-YesuiNeural"

    

    # Create TTS communicator

    communicate = edge_tts.Communicate(text, voice)

    

    # Save the speech to a file

    await communicate.save(filename)

    print(f"TTS saved as {filename}")



# Example usage

text_to_speak = "Сайн байна уу? Таныг харах сайхан байна."

asyncio.run(generate_mongolian_tts(text_to_speak))