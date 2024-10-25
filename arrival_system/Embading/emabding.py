import os
import json
import numpy as np  # NumPy is useful for calculating the average
from deepface import DeepFace

class FaceEmbeddingManager:
    def __init__(self, model_name, embedding_file="G:/firnass/New folder/arrivel_system/arrivel_system/arrival_system/Embading/embeddings.json"):
        self.model_name = model_name
        self.embedding_file = embedding_file

    def save_embeddings(self, embeddings, name):
        data = self._load_embeddings()
        # Overwrite the embedding for this name
        data[name] = embeddings
        with open(self.embedding_file, 'w') as file:
            json.dump(data, file, indent=4)

    def _load_embeddings(self):
        if os.path.exists(self.embedding_file):
            try:
                with open(self.embedding_file, 'r') as file:
                    return json.load(file)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def generate_embedding(self, img_path):
        """Generates an embedding for the given face image path."""
        try:
            face_embedding = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                enforce_detection=False
            )[0]["embedding"]
            return face_embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def calculate_average_embedding(self):
        """Calculates the average embedding from all saved embeddings."""
        embeddings_data = self._load_embeddings()
        if not embeddings_data:
            print("No embeddings found to calculate average.")
            return None

        all_embeddings = list(embeddings_data.values())
        # Convert list of embeddings to a NumPy array for easy averaging
        embeddings_array = np.array(all_embeddings)

        # Calculate the mean across embeddings
        average_embedding = np.mean(embeddings_array, axis=0)

        return average_embedding

# Usage example
if __name__ == "__main__":
    model_list = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
        "GhostFaceNet",
    ]
    model_name = model_list[1]  # Choose model
    face_embedding_manager = FaceEmbeddingManager(model_name)

    # Generate embedding for a face image
    face_region = "C://Users//edrie//Downloads//mes.jpeg"
    embedding = face_embedding_manager.generate_embedding(face_region)
    if embedding is not None:
        face_embedding_manager.save_embeddings(embedding, "person_name")

    # Calculate and print the average embedding from all saved embeddings
    avg_embedding = face_embedding_manager.calculate_average_embedding()
    if avg_embedding is not None:
        print("Average Embedding:", avg_embedding)
