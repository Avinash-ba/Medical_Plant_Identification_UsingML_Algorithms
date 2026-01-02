# Medicinal Plant Identification Using Machine Learning

This project is an AI-powered web application that can identify medicinal plants from images using deep learning. The system uses a Convolutional Neural Network (CNN) trained on plant images to classify different types of medicinal plants.

## Features

- **Image Upload**: Upload plant images through a user-friendly web interface
- **AI Prediction**: Uses TensorFlow/Keras CNN model for accurate plant identification
- **Plant Information**: Provides scientific names, health benefits, and traditional uses
- **Confidence Scoring**: Shows prediction confidence levels
- **Responsive Design**: Works on desktop and mobile devices

## Supported Plants

The model can identify the following medicinal plants:
- AMLA (Indian Gooseberry)
- BASIL (Holy Basil/Tulasi)
- NEEM (Azadirachta indica)
- GINGER (Zingiber officinale)
- TURMERIC (Curcuma longa)
- ALOEVERA (Aloe barbadensis)
- And many more traditional medicinal plants

## Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model (if needed)

If you don't have a pre-trained model, you can train one using your own dataset:

1. Organize your plant images in folders by plant type:
   ```
   dataset/
   ├── AMLA/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── BASIL/
   │   ├── image1.jpg
   │   └── ...
   └── ...
   ```

2. Run the training script:
   ```bash
   python train_model.py
   ```

### Step 2: Run the Web Application

Start the Flask web server:
```bash
python app.py
```

Open your browser and go to: `http://localhost:5000`

### Step 3: Use the Application

1. Click "Choose Plant Image" to select a plant photo
2. Click "Identify Plant" to get the prediction
3. View the results including plant name, confidence level, and medicinal information

## Project Structure

```
Medical_Plant_Identification_UsingML_Algorithms/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── index.html            # Web interface
├── requirements.txt       # Python dependencies
├── dataset/              # Training images (create this folder)
├── plant_identification_model.h5  # Trained model (generated)
├── class_indices.npy     # Class labels (generated)
└── README.md            # This file
```

## Model Architecture

The CNN model consists of:
- 4 Convolutional layers with ReLU activation
- MaxPooling layers for downsampling
- Dropout layer for regularization
- Dense layers for classification
- Softmax output for multi-class prediction

## Technologies Used

- **Backend**: Python Flask
- **AI/ML**: TensorFlow, Keras
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV, Pillow
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## API Endpoints

- `GET /`: Home page
- `POST /predict`: Plant identification (accepts image files)
- `GET /health`: Health check

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Disclaimer

This application is for educational and informational purposes only. Always consult healthcare professionals before using medicinal plants. Plant identification by AI should be verified by experts.

## Future Enhancements

- [ ] Mobile app development
- [ ] More plant species
- [ ] Real-time camera capture
- [ ] Plant care recommendations
- [ ] Offline model support
- [ ] Multi-language support
