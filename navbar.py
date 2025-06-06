import streamlit as st

st.set_page_config(page_title="My Portofolio")

st.sidebar.title('Info')
page = st.sidebar.radio('Pages:', ['Home', 'Project','About Me'])

if page == 'Home':
    st.markdown("<h1 style='text-align: center;'>Welcome To My Portofolio</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Data Science & Data Analyst Enthusiast</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Daniel Yosef Timisela</h2>", unsafe_allow_html=True)

elif page == 'Project':
    import project
    project.Project()

elif page == 'About Me':
    import about
    about.link()