# Video Search with CLIP

This Streamlit application allows users to upload a video, extract frames, and find objects or scenes based on text descriptions using OpenAI's CLIP (Contrastive Language-Image Pre-training) model.

## Features

- Upload video files in various formats (MP4, AVI, MOV, MKV)
- Extract frames from videos with configurable frame skipping
- Encode frames using the CLIP model
- Find frames that best match text queries
- Display results with timestamps and confidence scores

## Requirements

- Python 3.8+
- Hugging Face account and API token

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/video-object-detection.git
   cd video-object-detection
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Hugging Face token:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file and replace `your_hugging_face_token_here` with your actual Hugging Face token.

## Usage

1. Run the Streamlit application:

   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501).

3. Upload a video file using the file uploader.

4. Enter a text query describing what you want to find in the video.

5. View the best matching frame with its timestamp and confidence score.

## Testing

The project includes comprehensive test coverage:

1. Install test dependencies:

   ```
   pip install pytest mock pytest-mock pytest-cov
   ```

2. Run all tests:

   ```
   python tests/run_tests.py
   ```

   Or using pytest:

   ```
   pytest
   ```

3. Generate test coverage report:

   ```
   pytest --cov=app tests/
   ```

The test suite includes:

- Unit tests for individual functions
- Integration tests for the entire workflow
- UI tests for the Streamlit components
- Environment variable handling tests

See the `tests/README.md` file for more details on testing.

## How It Works

1. **Frame Extraction**: The application uses OpenCV to extract frames from the uploaded video at regular intervals.

2. **Frame Encoding**: Each extracted frame is resized to 224x224 pixels and encoded using the CLIP model to create embeddings.

3. **Text Query Processing**: The user's text query is also encoded using the CLIP model.

4. **Similarity Matching**: The application computes the cosine similarity between the text embedding and all frame embeddings.

5. **Result Display**: The frame with the highest similarity score is displayed along with its timestamp and confidence score.

## Limitations

- The application extracts frames at regular intervals (default: every 10 frames) to keep processing time reasonable.
- Very large videos may take longer to process.
- The accuracy of object detection depends on the CLIP model's capabilities.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
