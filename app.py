import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
import streamlit as st
import psycopg2
import plotly.graph_objects as go
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import DictCursor
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from sentence_transformers import SentenceTransformer
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem import EnumerateStereoisomers as ES
import faiss
from contextlib import contextmanager
import json
import gc
import psutil
import time
from threading import Lock
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings:
    """Application settings and configuration"""
    def __init__(self):
        self.MODEL_NAME = "all-MiniLM-L6-v2"
        self.DEVICE = "cpu"
        self.MAX_SEQ_LENGTH = 512
        self.MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        self.BATCH_SIZE = 100
        self.MAX_MEMORY = 500 * 1024 * 1024  # 500MB
        self.DB_CONFIG = {
            'host': os.getenv('PGHOST'),
            'database': os.getenv('PGDATABASE'),
            'user': os.getenv('PGUSER'),
            'password': os.getenv('PGPASSWORD'),
            'port': os.getenv('PGPORT')
        }
        self.API_TOKEN = os.getenv('API_TOKEN')

class ResourceManager:
    """Manages system resources and monitoring"""
    def __init__(self, threshold_mb: int = 1000):
        self.threshold_mb = threshold_mb
        self.measurements = []

    @contextmanager
    def monitor(self, operation_name: str):
        start_time = time.time()
        start_mem = self._get_memory_usage()
        try:
            yield
        finally:
            end_time = time.time()
            end_mem = self._get_memory_usage()
            self.measurements.append({
                'operation': operation_name,
                'duration': end_time - start_time,
                'memory_delta': end_mem - start_mem
            })
            if end_mem > self.threshold_mb:
                self._cleanup()

    def _get_memory_usage(self) -> float:
        return psutil.Process().memory_info().rss / 1024 / 1024

    def _cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class DatabaseManager:
    """Handles database operations"""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool = None
        self.setup_pool()

    def setup_pool(self):
        try:
            self.pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                **self.settings.DB_CONFIG
            )
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)

    def execute_query(self, query: str, params: tuple = None) -> Optional[List[Dict]]:
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, params)
                if cur.description:
                    return [dict(row) for row in cur.fetchall()]
                return None

    def save_processed_data(self, data: Dict[str, Any]) -> int:
        query = """
            INSERT INTO processed_data 
            (filename, content, embeddings, metadata)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """
        params = (
            data['filename'],
            data['content'],
            psycopg2.Binary(data['embeddings']),
            json.dumps(data['metadata'])
        )
        result = self.execute_query(query, params)
        return result[0]['id'] if result else None

class ModelManager:
    """Handles ML model operations"""
    _instance = None
    _lock = Lock()

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.index = None
        self._initialize_model()

    @classmethod
    def get_instance(cls, settings: Settings):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(settings)
        return cls._instance

    def _initialize_model(self):
        try:
            self.model = SentenceTransformer(self.settings.MODEL_NAME)
            self.model.eval()
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        try:
            return self.model.encode([text])[0]
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise

    def build_index(self, embeddings: np.ndarray):
        try:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        except Exception as e:
            logger.error(f"Index building error: {e}")
            raise

    def search(self, query: str, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        query_embedding = self.get_embedding(query).reshape(1, -1)
        return self.index.search(query_embedding, k)

class ChemicalProcessor:
    """Handles chemical data processing"""
    def __init__(self):
        self.resource_manager = ResourceManager()

    def process_smiles(self, smiles: str) -> Dict[str, Any]:
        """Process SMILES string and calculate molecular properties"""
        with self.resource_manager.monitor('smiles_processing'):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")

                properties = self._calculate_properties(mol)
                fingerprints = self._calculate_fingerprints(mol)
                stereochemistry = self._analyze_stereochemistry(mol)

                return {
                    'smiles': Chem.MolToSmiles(mol, canonical=True),
                    'inchi': Chem.MolToInchi(mol),
                    'inchi_key': Chem.MolToInchiKey(mol),
                    'properties': properties,
                    'fingerprints': fingerprints,
                    'stereochemistry': stereochemistry
                }
            except Exception as e:
                logger.error(f"Chemical processing error: {e}")
                raise

    def _calculate_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        return {
            'molecular_weight': Descriptors.ExactMolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'qed': Descriptors.qed(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol)
        }

    def _calculate_fingerprints(self, mol: Chem.Mol) -> Dict[str, Any]:
        return {
            'morgan': AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048),
            'maccs': AllChem.GetMACCSKeysFingerprint(mol),
            'topological': Chem.RDKFingerprint(mol)
        }

    def _analyze_stereochemistry(self, mol: Chem.Mol) -> Dict[str, Any]:
        return {
            'chiral_centers': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            'stereoisomers': len(list(ES.EnumerateStereoisomers(mol))),
            'has_stereochemistry': mol.HasSubstructMatch(Chem.MolFromSmarts('[C@H,C@@H,S@,S@@]'))
        }

class DrugDiscoveryApp:
    """Main application class"""
    def __init__(self):
        self.settings = Settings()
        self.db_manager = DatabaseManager(self.settings)
        self.model_manager = ModelManager.get_instance(self.settings)
        self.chemical_processor = ChemicalProcessor()
        self.resource_manager = ResourceManager()

    def setup_streamlit(self):
        st.set_page_config(
            page_title="Drug Discovery RAG System",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def run_streamlit(self):
        self.setup_streamlit()
        st.title("🧬 Drug Discovery RAG System")

        menu = ["Upload", "Search", "Analysis"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Upload":
            self.show_upload_page()
        elif choice == "Search":
            self.show_search_page()
        elif choice == "Analysis":
            self.show_analysis_page()

    def show_upload_page(self):
        st.header("📤 Upload Chemical Data")
        uploaded_file = st.file_uploader(
            "Upload file (CSV, Excel, PDF, Text)",
            type=['csv', 'xlsx', 'pdf', 'txt']
        )

        if uploaded_file:
            try:
                with self.resource_manager.monitor('file_processing'):
                    content = self.process_file(uploaded_file)
                    embeddings = self.model_manager.get_embedding(content)
                    
                    data = {
                        'filename': uploaded_file.name,
                        'content': content,
                        'embeddings': embeddings,
                        'metadata': {
                            'file_type': uploaded_file.type,
                            'size': uploaded_file.size
                        }
                    }
                    
                    self.db_manager.save_processed_data(data)
                    st.success("✅ File processed successfully!")
                    
                    with st.expander("View Content Preview"):
                        st.write(content[:500] + "...")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    def show_search_page(self):
        st.header("🔍 Search Chemical Database")
        query = st.text_input("Enter your search query:")
        k = st.slider("Number of results", 1, 10, 5)

        if query:
            try:
                with self.resource_manager.monitor('search'):
                    distances, indices = self.model_manager.search(query, k)
                    
                    st.subheader("Search Results")
                    self.display_search_results(distances[0], indices[0])
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")

    def show_analysis_page(self):
        st.header("🧪 Chemical Analysis")
        smiles = st.text_input("Enter SMILES string:")

        if smiles:
            try:
                with self.resource_manager.monitor('analysis'):
                    result = self.chemical_processor.process_smiles(smiles)
                    self.display_chemical_analysis(result)
                    
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

    def display_search_results(self, distances: np.ndarray, indices: np.ndarray):
        for i, (distance, idx) in enumerate(zip(distances, indices)):
            confidence = 1 / (1 + distance)
            st.write(f"Result {i+1} (Confidence: {confidence:.2f})")
            
            data = self.db_manager.execute_query(
                "SELECT * FROM processed_data WHERE id = %s",
                (idx,)
            )[0]
            
            with st.expander("View Details"):
                st.write(data['content'][:500] + "...")
                st.json(data['metadata'])

    def display_chemical_analysis(self, result: Dict[str, Any]):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            st.write(f"SMILES: {result['smiles']}")
            st.write(f"InChI: {result['inchi']}")
            st.write(f"InChI Key: {result['inchi_key']}")

        with col2:
            st.subheader("Properties")
            for name, value in result['properties'].items():
                st.write(f"{name}: {value:.2f}")

        st.subheader("Stereochemistry")
        st.json(result['stereochemistry'])

    def process_file(self, file) -> str:
        """Process uploaded file and extract content"""
        if file.size > self.settings.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds limit ({self.settings.MAX_FILE_SIZE/1024/1024}MB)")
            
        content = None
        try:
            if file.type == 'text/csv':
                df = pd.read_csv(file)
                content = df.to_string()
            elif file.type == 'application/pdf':
                # Add PDF processing logic
                pass
            elif file.type == 'text/plain':
                content = file.getvalue().decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file.type}")
                
            return content
        except Exception as e:
            logger.error(f"File processing error: {e}")
            raise

def main():
    """Main entry point"""
    try:
        app = DrugDiscoveryApp()
        app.run_streamlit()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()