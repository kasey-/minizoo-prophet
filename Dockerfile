FROM conda/miniconda3

COPY ./prophet /app
WORKDIR /app

RUN conda install --yes -c conda-forge fbprophet
RUN conda install --yes --file requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

CMD [ "gunicorn", "--workers=4", "--timeout=120", "-b 0.0.0.0:80", "main:app" ]