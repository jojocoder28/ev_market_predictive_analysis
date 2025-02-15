# EV Marketing and Clustering Analysis

## Overview
This Streamlit application provides a comprehensive platform for analyzing Electric Vehicle (EV) adoption trends, customer segmentation, and campaign targeting. It leverages advanced machine learning models, clustering techniques, and Google's Gemini Generative AI to deliver actionable insights and recommendations.

## Features
1. **Campaign Prediction**
   - Predicts marketing campaign targeting for individual users based on their demographics, preferences, and financial profiles.
   - Provides AI-generated feedback and business recommendations using Gemini Generative AI.

2. **Customer Clustering**
   - Groups customers into clusters using K-Means clustering for targeted marketing and insights.
   - Offers AI-generated summaries and recommendations for each customer segment.

3. **EV Adoption Prediction**
   - Analyzes regional EV adoption rates based on various factors.
   - Uses Gemini Generative AI to predict adoption rates and provide detailed insights.

## Installation
Follow these steps to set up and run the application:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ev-marketing-analysis.git
   cd ev-marketing-analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your environment variables:
   - Create a `.env` file in the root directory and add your API key:
     ```
     API_KEY=your_genai_api_key
     ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## File Structure
```
.
├── app.py                 # Main application file
├── dataset_individual.csv # Dataset for customer clustering
├── dataset_regional.csv   # Dataset for EV adoption prediction
├── final_model            # Trained machine learning model for campaign prediction
├── label_encoders.pkl     # Encoders for categorical data
├── requirements.txt       # Required Python dependencies
├── .env                   # Environment variables (not included in repo)
├── README.md              # Documentation
```

## Dependencies
- **Python 3.8+**
- **Libraries**:
  - Streamlit
  - Pandas
  - NumPy
  - scikit-learn
  - Matplotlib
  - Seaborn
  - PyCaret
  - Joblib
  - Python-dotenv
  - Google Generative AI SDK

## Usage
### 1. Campaign Prediction
- Navigate to the **Campaign Prediction** page via the sidebar.
- Input user information such as age, annual income, and preferences.
- Click **Predict Campaign Targeting** to get predictions and AI-generated recommendations.

### 2. Customer Clustering
- Navigate to the **Customer Clustering** page.
- View clustering results based on customer profiles.
- Review AI-generated summaries and recommendations for each cluster.

### 3. EV Adoption Prediction
- Navigate to the **EV Adoption Prediction** page.
- Input regional data for factors influencing EV adoption.
- Click **Predict with Gemini** to analyze adoption rates and receive insights.

## Key Technologies
1. **Streamlit**: Provides an intuitive user interface for data input and visualization.
2. **K-Means Clustering**: Groups customers into distinct clusters for targeted marketing.
3. **Google Gemini AI**: Generates business recommendations and predicts outcomes using generative AI.
4. **PyCaret**: Simplifies machine learning model deployment and predictions.

## Datasets
- **Individual Customer Dataset**: Contains customer demographic and preference data.
- **Regional Dataset**: Contains regional statistics and competitor analysis for EV adoption prediction.

## Contribution
Contributions are welcome! Please create a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Google's Gemini AI for generative insights.
- Open-source libraries and frameworks for enabling rapid development.

# ev_market_predictive_analysis
