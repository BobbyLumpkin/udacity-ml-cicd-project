# Pull pandas python3.8 container.
FROM amancevice/pandas:1.1.1

# Set environment vars, working dir and expose port.
ENV PYTHONUNBUFFERED 1 
EXPOSE 8501
WORKDIR /app

# Copy project and install dependencies from requirements.txt.
COPY . .
RUN pip install -r requirements.txt

# Run the application on port 8501.
CMD streamlit run streamlit_frontend.py --server.port 8501