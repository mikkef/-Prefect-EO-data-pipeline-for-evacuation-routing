FROM continuumio/miniconda3

ARG CONDA_ENV_NAME=embedding_based_surface_classification_env

WORKDIR /app

# environment.yml: export with
#   conda env export --from-history -n <env> > environment.yml
# then delete the "prefix:" line and make sure prefect, boto3, geopandas, osmnx are listed.
COPY environment.yml .
RUN conda env create -n ${CONDA_ENV_NAME} -f environment.yml && conda clean -afy

# Use the env's python by default (also what sys.executable returns in subprocesses)
ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

COPY src ./src

# APP_ENTRYPOINT is set in docker-compose.yml
CMD ["sh", "-c", "python \"$APP_ENTRYPOINT\""]
