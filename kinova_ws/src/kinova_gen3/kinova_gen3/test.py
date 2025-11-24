from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dOXf27URLjdeZMgyJ7en"
)

try:
    result = CLIENT.infer("undistorted_image.jpg", model_id="cube-color-gzmh4/14")
    print("Success!")
    print(result)
except Exception as e:
    print(f"Error: {e}")