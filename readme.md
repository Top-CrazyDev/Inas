1. Install the required dependencies using the following command:

    ```bash
    pip install -r requirements.txt
    ```

2. Open the `prompt.json` file and set the prompt parameters:
    - `artist`: Set this to "eclips".
    - `dimension`: Specify the desired dimensions.
    - `color`: Set the color for the image.

    Example `prompt.json` content:
    ```json
    {
      "artist": "eclips",
      "dimension": [800 , 600],
      "color": [255 , 255 , 255]
    }
    ```

3. Execute the following command to run the application:

    ```bash
    python app.py
    ```
