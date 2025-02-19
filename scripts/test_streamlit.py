import streamlit as st
import plotly.express as px

# Create your plot
fig = px.line(x=[1, 2, 3], y=[1, 2, 3])

# Display in Streamlit
st.plotly_chart(fig)