from gradio_client import Client
import shutil

client = Client("NihalGazi/Text-To-Speech-Unlimited")
result = client.predict(
		prompt="Hi there! How can I help you.",
		voice="alloy",
		emotion="Sad",
		use_random_seed=True,
		specific_seed=12345,
		api_name="/text_to_speech_app"
)
shutil.move(result[0], "output.wav")   # save it permanently
print(result)