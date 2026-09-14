# Dockerfile
FROM python:3.12-slim
# Éviter Alpine : problèmes de compatibilité avec numpy/pandas

WORKDIR /app


# Étape 1 : copier les fichiers dont pip a besoin pour installer le package
# -> pyproject.toml décrit le projet
# -> README.md est requis car pyproject.toml le référence (readme = "README.md")
# -> src/ contient le code du package que setuptools doit découvrir
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Étape 2 : installer le package et ses dépendances
RUN pip install --no-cache-dir .

# Étape 3 : copier le reste du projet (config, main.py, etc.)
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
