import streamlit as st
import pandas as pd
from htmlTemplates import css
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="investment-advisor/chat", layout="wide")
vectorizer = TfidfVectorizer()

def get_csv_content(csv_docs):
    # Initialize an empty DataFrame
    data = pd.DataFrame()

    # Loop through each CSV path and concatenate the content
    for csv in csv_docs:
        try:
            # Read the content of each CSV
            csv_content = pd.read_csv(csv, delimiter=';')
            # Concatenate to the existing DataFrame
            data = pd.concat([data, csv_content], ignore_index=True)
        except Exception as e:
            print(f"Error processing file {csv}: {e}")
    
    return data
  
def get_vector_store(csv_content):
    # Check if the content is read as a DataFrame
    print(csv_content.head())

    # Create the 'input_features' column by concatenating 'perfil' and 'Valor de aplicação inicial'
    csv_content['input_features'] = csv_content['profile'] + ' ' + str(csv_content['initial_investment_amount']) + ' ' + csv_content['initial_period']

    # Vectorize the input features
    vector = vectorizer.fit_transform(csv_content['input_features'])
    return vector

def recommend_products(profile, period, value, vector, data, num_products):
    input_user = f'{profile} {value} {period}'
    input_vector = vectorizer.transform([input_user])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(input_vector, vector)

    # Ensure num_products is an integer
    if not isinstance(num_products, int):
        raise TypeError("The 'num_products' parameter must be an integer.")
    
    # Sort and get the indices of the most similar products
    indexes = similarities.argsort().flatten()[-num_products:]
    
    recommendations = data.iloc[indexes]
    
    # Generate recommendations in a descriptive text format
    text_recommendation = f"Based on your profile '{profile}', investment amount of R${value:,.2f}, and period '{period}', we recommend the following products:\n"
    
    # Calculate value allocations based on similarities
    recommended_similarity = similarities.flatten()[indexes]
    sum_similarity = recommended_similarity.sum()

    # Store all products in a list
    recommended_products = []
    for i in range(len(indexes)):  # Loop through the most similar recommendations
        for j in range(1, 4):  # Iterate over products 1, 2, and 3
            if f'product_{j}_family' in recommendations.columns and f'product_{j}_id' in recommendations.columns:
                product_family = recommendations.iloc[i][f'product_{j}_family']
                product_identifier = recommendations.iloc[i][f'product_{j}_id']
                recommended_products.append((product_family, product_identifier))
    
    # Reduce the list to the requested number of products
    recommended_products = recommended_products[:num_products]

    # Calculate value distributed proportionally
    for i, (product_family, product_identifier) in enumerate(recommended_products):
        similarity_percentage = recommended_similarity[i % len(recommended_similarity)] / sum_similarity
        value_per_product = similarity_percentage * value
        
        # Add the product to the description
        text_recommendation += (
            f"- Product {i + 1}: {product_family} (Identifier: {product_identifier})\n"
            f"  - Value: R${value_per_product:,.2f} (Distribution: {similarity_percentage * 100:.2f}%)\n"
        )
    
    return text_recommendation

def main():
  
    # Apply CSS
    st.write(css, unsafe_allow_html=True)
  
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
  
    # Sidebar with button to redirect
    st.title("Investment Advisor")

    # User input for profile selection
    profile = st.selectbox("Select your profile:",  ["Ultra-Conservative", "Conservative", "Moderate", "Dynamic"])

    # User input for investment amount
    value = st.number_input("Enter the investment amount:", min_value=0.0, max_value=300000.0, step=500.0)

    # User input for number of recommended products
    num_products = st.number_input("Enter the number of recommended products you wish to receive:", min_value=1, max_value=5, step=2)

    # User input for period selection
    period = st.selectbox("Select the period:", ["Menos que 6 meses", "6 meses a 1 ano", "Mais que 1 ano"])

    recommendations = ''
  
    # Sidebar with button to redirect
    with st.sidebar:
        st.subheader('Your Documents')
        csv_docs = st.file_uploader("Upload your files here, and click 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get CSV documents
                csv_content = get_csv_content(csv_docs)
                
                # Create vector store with embeddings
                vector_store = get_vector_store(csv_content)   
                
                recommendations = recommend_products(profile, period, value, vector_store, csv_content, num_products)
                # Create conversation chain
                
    st.divider()
        
    st.subheader("Product Recommendations:")
    
    if not recommendations:
        st.write("No recommendations available at the moment.")
    else:
        st.write(recommendations)

if __name__ == '__main__':
    main()
