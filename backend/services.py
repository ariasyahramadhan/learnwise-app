import io
import re
import numpy as np
from itertools import combinations
from typing import List, Dict, Any
from collections import Counter

import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap


class TextExtractor:
    """Ekstrak teks dari berbagai format file"""

    def extract(self, content: bytes, content_type: str) -> str:
        if content_type == "application/pdf":
            return self._extract_pdf(content)
        elif "wordprocessingml" in content_type:
            return self._extract_docx(content)
        elif content_type == "text/plain":
            return content.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Tipe file tidak didukung: {content_type}")

    def _extract_pdf(self, content: bytes) -> str:
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return self._clean_text(text)

    def _extract_docx(self, content: bytes) -> str:
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text


class ModelASimilarity:
    """
    Deteksi kemiripan menggunakan pipeline Model A (XGBoost + TF-IDF)
    dari notebook LearnWise — Surface Similarity (Deteksi Plagiarisme).

    Membutuhkan file-file berikut di folder 'model_artifacts/':
        - tfidf_vectorizer.pkl
        - tfidf_vectorizer_char.pkl
        - xgboost_classifier.pkl
        - pipeline_config.pkl
    """

    ARTIFACTS_PATH = "model_artifacts"

    def __init__(self):
        self._vectorizer = None
        self._vectorizer_char = None
        self._clf = None
        self._config = None
        self._stemmer = None
        self._loaded = False
        self._load_error = None
        self._try_load()

    def _try_load(self):
        """Muat model dari disk. Gagal secara diam-diam agar server tetap bisa jalan."""
        try:
            import joblib
            import os

            vect_path      = os.path.join(self.ARTIFACTS_PATH, "surface_similarity", "surface_word_vectorizer.pkl")
            vect_char_path = os.path.join(self.ARTIFACTS_PATH, "surface_similarity", "surface_char_vectorizer.pkl")
            clf_path       = os.path.join(self.ARTIFACTS_PATH, "surface_similarity", "surface_classifier.pkl")
            config_path    = os.path.join(self.ARTIFACTS_PATH, "surface_similarity", "surface_config.pkl")

            missing = [p for p in [vect_path, vect_char_path, clf_path, config_path] if not os.path.exists(p)]
            if missing:
                self._load_error = (
                    f"File model_artifacts tidak ditemukan: {missing}. "
                    "Pastikan model hasil training Model A (surface_similarity) sudah disalin "
                    "ke dalam folder backend/model_artifacts/surface_similarity/."
                )
                print(f"[ModelA] WARNING: {self._load_error}")
                return

            self._vectorizer      = joblib.load(vect_path)
            self._vectorizer_char = joblib.load(vect_char_path)
            self._clf             = joblib.load(clf_path)
            self._config          = joblib.load(config_path)
            self._threshold       = self._config.get("threshold_optimal", 0.5)

            # Sastrawi stemmer — opsional, fallback ke tanpa stemming jika tidak ada
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                factory = StemmerFactory()
                self._stemmer = factory.create_stemmer()
            except ImportError:
                print("[ModelA] INFO: PySastrawi tidak tersedia, preprocessing tanpa stemming.")

            self._loaded = True
            print(f"[ModelA] Pipeline loaded. Threshold optimal: {self._threshold:.4f}")

        except Exception as e:
            self._load_error = str(e)
            print(f"[ModelA] ERROR saat memuat model: {e}")

    @property
    def is_available(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    # Preprocessing — sama persis dengan notebook                         #
    # ------------------------------------------------------------------ #

    def _preprocess(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if self._stemmer:
            text = self._stemmer.stem(text)
        return text

    # ------------------------------------------------------------------ #
    # 26 feature extractors — sama persis dengan notebook                 #
    # ------------------------------------------------------------------ #

    def _jaccard(self, s1: str, s2: str) -> float:
        w1, w2 = set(s1.split()), set(s2.split())
        if not w1 and not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def _ngram_overlap(self, s1: str, s2: str, n: int) -> float:
        def get_ng(t):
            words = t.split()
            return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set()
        ng1, ng2 = get_ng(s1), get_ng(s2)
        if not ng1 or not ng2:
            return 0.0
        return len(ng1 & ng2) / max(len(ng1), len(ng2))

    def _positional_ngram(self, s1: str, s2: str, n: int) -> float:
        w1, w2 = s1.split(), s2.split()
        if len(w1) < n or len(w2) < n:
            return 0.0
        bg1 = [tuple(w1[i:i+n]) for i in range(len(w1)-n+1)]
        bg2 = [tuple(w2[i:i+n]) for i in range(len(w2)-n+1)]
        return sum(1 for i, b in enumerate(bg1) if i < len(bg2) and b == bg2[i]) / max(len(bg1), len(bg2))

    def _word_order_score(self, s1: str, s2: str) -> float:
        w1, w2 = s1.split(), s2.split()
        common = set(w1) & set(w2)
        if not common:
            return 0.5
        pos1 = {w: i for i, w in enumerate(w1) if w in common}
        pos2 = {w: i for i, w in enumerate(w2) if w in common}
        words = list(common)[:30]
        con = dis = 0
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                wi, wj = words[i], words[j]
                if all(k in pos1 and k in pos2 for k in [wi, wj]):
                    d1 = pos1[wi] - pos1[wj]
                    d2 = pos2[wi] - pos2[wj]
                    if d1 * d2 > 0:
                        con += 1
                    elif d1 * d2 < 0:
                        dis += 1
        return con / (con + dis) if (con + dis) > 0 else 0.5

    def _lcs_ratio(self, s1: str, s2: str) -> float:
        w1, w2 = s1.split()[:50], s2.split()[:50]
        m, n = len(w1), len(w2)
        if not m or not n:
            return 0.0
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = dp[i-1][j-1]+1 if w1[i-1]==w2[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n] / max(m, n)

    def _edit_distance_ratio(self, s1: str, s2: str) -> float:
        a, b = s1[:200], s2[:200]
        m, n = len(a), len(b)
        if not m and not n:
            return 1.0
        dp = list(range(n+1))
        for i in range(1, m+1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, n+1):
                dp[j] = prev[j-1] if a[i-1]==b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
        return 1.0 - dp[n] / max(m, n)

    def _char_ngram_jaccard(self, s1: str, s2: str, n: int) -> float:
        def get_cng(t):
            return set(t[i:i+n] for i in range(len(t)-n+1)) if len(t) >= n else set()
        c1, c2 = get_cng(s1), get_cng(s2)
        if not c1 or not c2:
            return 0.0
        return len(c1 & c2) / len(c1 | c2)

    def _prefix_match_ratio(self, s1: str, s2: str, pl: int = 3) -> float:
        w1, w2 = s1.split(), s2.split()
        if not w1 or not w2:
            return 0.0
        p1 = set(w[:pl] for w in w1 if len(w) >= pl)
        p2 = set(w[:pl] for w in w2 if len(w) >= pl)
        if not p1 or not p2:
            return 0.0
        return len(p1 & p2) / len(p1 | p2)

    def _token_sort_cosine(self, s1: str, s2: str) -> float:
        ss1 = ' '.join(sorted(s1.split()))
        ss2 = ' '.join(sorted(s2.split()))
        v1 = self._vectorizer.transform([ss1])
        v2 = self._vectorizer.transform([ss2])
        from sklearn.metrics.pairwise import paired_cosine_distances
        return float(1.0 - paired_cosine_distances(v1, v2)[0])

    def _extract_features(self, s1_clean: str, s2_clean: str) -> np.ndarray:
        """Ekstrak 26 fitur — urutan harus sama persis dengan saat training."""
        from sklearn.metrics.pairwise import paired_cosine_distances

        # 1. cosine_word
        v1_word = self._vectorizer.transform([s1_clean])
        v2_word = self._vectorizer.transform([s2_clean])
        cw = float(1.0 - paired_cosine_distances(v1_word, v2_word)[0])

        # 2. cosine_char
        v1_char = self._vectorizer_char.transform([s1_clean])
        v2_char = self._vectorizer_char.transform([s2_clean])
        cc = float(1.0 - paired_cosine_distances(v1_char, v2_char)[0])

        # 3. jaccard
        jacc = self._jaccard(s1_clean, s2_clean)

        # 4. bigram
        bi = self._ngram_overlap(s1_clean, s2_clean, 2)

        # 5. trigram
        tri = self._ngram_overlap(s1_clean, s2_clean, 3)

        # 6. fourgram
        fg = self._ngram_overlap(s1_clean, s2_clean, 4)

        # 7. pos_bigram
        pbi = self._positional_ngram(s1_clean, s2_clean, 2)

        # 8. pos_trigram
        ptri = self._positional_ngram(s1_clean, s2_clean, 3)

        # 9. word_order
        wo = self._word_order_score(s1_clean, s2_clean)

        # 10. lcs
        lcs = self._lcs_ratio(s1_clean, s2_clean)

        # 11. edit_dist
        ed = self._edit_distance_ratio(s1_clean, s2_clean)

        # 12. char_bi_j
        cbj = self._char_ngram_jaccard(s1_clean, s2_clean, 2)

        # 13. char_tri_j
        ctj = self._char_ngram_jaccard(s1_clean, s2_clean, 3)

        # 14. prefix_match
        pfx = self._prefix_match_ratio(s1_clean, s2_clean)

        # 15. token_sort_cosine
        tsc = self._token_sort_cosine(s1_clean, s2_clean)

        # Length variables
        w1, w2 = s1_clean.split(), s2_clean.split()
        ls1 = float(len(w1))
        ls2 = float(len(w2))

        # 16. len_diff
        ldiff = abs(ls1 - ls2)

        # 17. len_ratio
        lratio = min(ls1, ls2) / max(max(ls1, ls2), 1.0)

        # 18. uniq_ratio
        set1, set2 = set(w1), set(w2)
        uniq = len(set1 ^ set2) / max(len(set1 | set2), 1.0)

        # 19. shared_count
        shared = float(len(set1 & set2))

        # 20. len_s1
        # ls1 (20)

        # 21. len_s2
        # ls2 (21)

        # Cross-features
        # 22. cosine_prod
        cp = cw * cc

        # 23. jacc_x_order
        jxo = jacc * wo

        # 24. mean_cosine
        mc = (cw + cc) / 2.0

        # 25. overlap_ratio
        ovr = shared / max(ls1 + ls2, 1.0)

        # 26. abs_len_norm
        aln = ldiff / max(ls1 + ls2, 1.0)

        return np.array([[
            cw, cc, jacc, bi, tri, fg, pbi, ptri, wo, lcs,
            ed, cbj, ctj, pfx, tsc,
            ldiff, lratio, uniq, shared, ls1, ls2,
            cp, jxo, mc, ovr, aln
        ]], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Kembalikan probabilitas kemiripan (0.0–1.0) antara dua teks.
        Nilai sudah dinormalisasi ke rentang yang sebanding dengan skor
        model lain di sistem ini (semakin tinggi = semakin mirip).
        """
        s1 = self._preprocess(text_a)
        s2 = self._preprocess(text_b)
        feats = self._extract_features(s1, s2)
        prob  = float(self._clf.predict_proba(feats)[0][1])
        return prob

    def compute_matrix(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        matrix = np.eye(n)
        for i, j in combinations(range(n), 2):
            sim = self.similarity(texts[i], texts[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
        return matrix


class ModelBSimilarity:
    """
    Deteksi kemiripan menggunakan pipeline Model B (Stylometric Similarity)
    dari notebook Model B ver1.ipynb.

    Membutuhkan file-file berikut di folder 'model_artifacts/':
        - xgboost_classifier_b.pkl
        - pipeline_config_b.pkl
    """

    ARTIFACTS_PATH = "model_artifacts"

    def __init__(self):
        self._clf = None
        self._config = None
        self._loaded = False
        self._load_error = None
        self._try_load()

    def _try_load(self):
        """Muat model dari disk. Gagal secara diam-diam agar server tetap bisa jalan."""
        try:
            import joblib
            import os

            clf_path    = os.path.join(self.ARTIFACTS_PATH, "stylometric_similarity", "stylometric_classifier.pkl")
            config_path = os.path.join(self.ARTIFACTS_PATH, "stylometric_similarity", "stylometric_config.pkl")

            missing = [p for p in [clf_path, config_path] if not os.path.exists(p)]
            if missing:
                self._load_error = (
                    f"File model_artifacts b tidak ditemukan: {missing}. "
                    "Pastikan model hasil training Model B (stylometric_similarity) sudah disalin "
                    "ke dalam folder backend/model_artifacts/stylometric_similarity/."
                )
                print(f"[ModelB] WARNING: {self._load_error}")
                return

            self._clf        = joblib.load(clf_path)
            self._config     = joblib.load(config_path)
            self._threshold  = self._config.get("threshold_optimal", 0.5)
            self._loaded = True
            print(f"[ModelB] Pipeline loaded. Threshold optimal: {self._threshold:.4f}")

        except Exception as e:
            self._load_error = str(e)
            print(f"[ModelB] ERROR saat memuat model: {e}")

    @property
    def is_available(self) -> bool:
        return self._loaded

    def _get_stylometry_vector(self, text: str) -> np.ndarray:
        text_str = str(text)
        words = text_str.split()
        total_words = len(words)
        total_chars = len(text_str)
        
        if total_words == 0:
            return np.zeros(16)
            
        avg_word_len = sum(len(w) for w in words) / total_words
        
        # Split sentences by . ! or ?
        sentences = re.split(r'[.!?]+', text_str)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sent_len = total_words / max(len(sentences), 1)
        
        unique_words = len(set(w.lower() for w in words))
        ttr = unique_words / total_words
        
        comma_freq = text_str.count(',') / max(total_chars, 1)
        period_freq = text_str.count('.') / max(total_chars, 1)
        colon_freq = text_str.count(':') / max(total_chars, 1)
        dash_freq = (text_str.count('-') + text_str.count('_')) / max(total_chars, 1)
        question_freq = text_str.count('?') / max(total_chars, 1)
        
        text_lower = text_str.lower()
        stopword_yang = text_lower.count('yang') / total_words
        stopword_dan = text_lower.count('dan') / total_words
        stopword_di = text_lower.count('di') / total_words
        stopword_ke = text_lower.count('ke') / total_words
        stopword_dari = text_lower.count('dari') / total_words
        stopword_untuk = text_lower.count('untuk') / total_words
        
        uppercase_ratio = sum(1 for c in text_str if c.isupper()) / max(total_chars, 1)
        digit_ratio = sum(1 for c in text_str if c.isdigit()) / max(total_chars, 1)
        
        return np.array([
            avg_word_len, avg_sent_len, ttr,
            comma_freq, period_freq, colon_freq, dash_freq, question_freq,
            stopword_yang, stopword_dan, stopword_di, stopword_ke, stopword_dari, stopword_untuk,
            uppercase_ratio, digit_ratio
        ])

    def _extract_features(self, s1: str, s2: str) -> np.ndarray:
        v1 = self._get_stylometry_vector(s1)
        v2 = self._get_stylometry_vector(s2)
        
        diff = np.abs(v1 - v2)
        mean = (v1 + v2) / 2.0
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 > 0 and norm_v2 > 0:
            cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
        else:
            cosine = 0.0
            
        corr = np.corrcoef(v1, v2)[0, 1] if norm_v1 > 0 and norm_v2 > 0 else 0.0
        if np.isnan(corr):
            corr = 0.0
            
        return np.array([np.hstack([diff, mean, [cosine, corr]])], dtype=np.float32)

    def similarity(self, text_a: str, text_b: str) -> float:
        feats = self._extract_features(text_a, text_b)
        prob  = float(self._clf.predict_proba(feats)[0][1])
        return prob

    def compute_matrix(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        matrix = np.eye(n)
        for i, j in combinations(range(n), 2):
            sim = self.similarity(texts[i], texts[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
        return matrix


class NGramSimilarity:
    """Deteksi plagiarisme berdasarkan kecocokan susunan kata (n-gram overlap)"""

    def __init__(self, n: int = 3):
        self.n = n

    def get_ngrams(self, text: str) -> Counter:
        words = re.findall(r'\b\w+\b', text.lower())
        ngrams = [tuple(words[i:i+self.n]) for i in range(len(words) - self.n + 1)]
        return Counter(ngrams)

    def similarity(self, text_a: str, text_b: str) -> float:
        ngrams_a = self.get_ngrams(text_a)
        ngrams_b = self.get_ngrams(text_b)

        if not ngrams_a or not ngrams_b:
            return 0.0

        set_a = set(ngrams_a.keys())
        set_b = set(ngrams_b.keys())
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        if union == 0:
            return 0.0

        return intersection / union

    def compute_matrix(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        matrix = np.eye(n)
        for i, j in combinations(range(n), 2):
            sim = self.similarity(texts[i], texts[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
        return matrix


class PlagiarismService:
    def __init__(self):
        print("Loading SBERT model...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.extractor = TextExtractor()
        self.ngram = NGramSimilarity(n=3)
        self.model_a = ModelASimilarity()
        self.model_b = ModelBSimilarity()

        if self.model_a.is_available:
            print("Model A (XGBoost pipeline) loaded successfully!")
        else:
            print(f"Model A tidak tersedia: {self.model_a._load_error}")
            print("Model 'model_a' akan fallback ke NGram jika dipilih.")

        if self.model_b.is_available:
            print("Model B (Stylometry pipeline) loaded successfully!")
        else:
            print(f"Model B tidak tersedia: {self.model_b._load_error}")
            print("Model 'model_b' akan fallback ke NGram jika dipilih.")

        print("SBERT loaded!")

    def analyze(self, file_contents: List[Dict], model: str = "model_a", weight_a: float = 0.5, weight_b: float = 0.5) -> Dict[str, Any]:
        # 1. Ekstrak teks
        documents = []
        for fc in file_contents:
            text = self.extractor.extract(fc["content"], fc["content_type"])
            if not text or len(text.strip()) < 10:
                raise ValueError(f"File '{fc['name']}' tidak memiliki teks yang dapat dibaca")
            documents.append({
                "name": fc["name"],
                "text": text,
                "preview": text[:200] + "..." if len(text) > 200 else text
            })

        texts = [doc["text"] for doc in documents]

        # 2. Hitung similarity sesuai model
        if model == "model_b":
            if self.model_b.is_available:
                similarity_matrix = self.model_b.compute_matrix(texts)
            else:
                print("[WARNING] Model B tidak tersedia, fallback ke NGram.")
                similarity_matrix = self.ngram.compute_matrix(texts)
        elif model == "model_c":
            if self.model_a.is_available:
                matrix_a = self.model_a.compute_matrix(texts)
            else:
                matrix_a = self.ngram.compute_matrix(texts)

            if self.model_b.is_available:
                matrix_b = self.model_b.compute_matrix(texts)
            else:
                matrix_b = self.ngram.compute_matrix(texts)

            similarity_matrix = weight_a * matrix_a + weight_b * matrix_b
        else:  # model_a
            if self.model_a.is_available:
                similarity_matrix = self.model_a.compute_matrix(texts)
            else:
                print("[WARNING] Model A tidak tersedia, fallback ke NGram.")
                similarity_matrix = self.ngram.compute_matrix(texts)

        # 3. Hitung pasangan
        pairs = []
        for i, j in combinations(range(len(documents)), 2):
            similarity_score   = float(similarity_matrix[i][j])
            similarity_percent = round(similarity_score * 100, 2)

            if similarity_percent >= 80:
                risk = "high"
            elif similarity_percent >= 50:
                risk = "medium"
            else:
                risk = "low"

            pairs.append({
                "doc_a": documents[i]["name"],
                "doc_b": documents[j]["name"],
                "doc_a_index": i,
                "doc_b_index": j,
                "similarity": similarity_percent,
                "risk": risk
            })

        pairs.sort(key=lambda x: x["similarity"], reverse=True)

        # 4. Reduksi dimensi — selalu pakai SBERT embeddings untuk visualisasi
        sbert_embeddings = self.model.encode(texts, convert_to_numpy=True)
        coordinates_2d   = self._reduce_dimensions(sbert_embeddings)

        # 5. Buat matrix
        matrix = []
        for i in range(len(documents)):
            row = []
            for j in range(len(documents)):
                row.append(round(float(similarity_matrix[i][j]) * 100, 2))
            matrix.append(row)

        return {
            "documents": [
                {
                    "index": i,
                    "name": doc["name"],
                    "preview": doc["preview"],
                    "text": doc["text"],
                    "word_count": len(doc["text"].split())
                }
                for i, doc in enumerate(documents)
            ],
            "pairs": pairs,
            "similarity_matrix": matrix,
            "coordinates_2d": coordinates_2d,
            "model": model,
            "model_a_available": self.model_a.is_available,
            "model_b_available": self.model_b.is_available,
            "summary": {
                "total_documents": len(documents),
                "total_pairs": len(pairs),
                "highest_similarity": pairs[0]["similarity"] if pairs else 0,
                "average_similarity": round(
                    sum(p["similarity"] for p in pairs) / len(pairs), 2
                ) if pairs else 0
            }
        }

    def _sbert_similarity(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return cosine_similarity(embeddings)

    def _reduce_dimensions(self, embeddings: np.ndarray) -> List[Dict]:
        n = len(embeddings)

        if n <= 2:
            coords = np.array([[0.0, 0.0], [1.0, 0.0]])
            if n == 1:
                coords = np.array([[0.0, 0.0]])
        elif n <= 4:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)
        else:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(n - 1, 15),
                min_dist=0.3,
                random_state=42
            )
            coords = reducer.fit_transform(embeddings)

        for dim in range(coords.shape[1]):
            col = coords[:, dim]
            col_range = col.max() - col.min()
            if col_range > 0:
                coords[:, dim] = (col - col.min()) / col_range * 2 - 1

        return [
            {"x": float(coords[i][0]), "y": float(coords[i][1])}
            for i in range(n)
        ]


class EssayScoringService:
    """
    Evaluasi nilai esai berdasarkan kunci jawaban (reference answer)
    menggunakan model Gradient Boosting Regressor dan StandardScaler.
    """

    def __init__(self, sbert_model):
        self.sbert_model = sbert_model
        self._model = None
        self._scaler = None
        self._loaded = False
        self._load_error = None
        self._try_load()

    def _try_load(self):
        """Muat model dan scaler dari disk."""
        try:
            import joblib
            import os

            model_path = os.path.join("model_artifacts", "essay_scoring", "scoring_model.pkl")
            scaler_path = os.path.join("model_artifacts", "essay_scoring", "scaler.pkl")

            # Fallback path jika dijalankan dari root directory proyek
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                alt_model_path = os.path.join("backend", "model_artifacts", "essay_scoring", "scoring_model.pkl")
                alt_scaler_path = os.path.join("backend", "model_artifacts", "essay_scoring", "scaler.pkl")
                if os.path.exists(alt_model_path) and os.path.exists(alt_scaler_path):
                    model_path = alt_model_path
                    scaler_path = alt_scaler_path
                else:
                    self._load_error = (
                        "File model essay_scoring tidak ditemukan di model_artifacts/essay_scoring/."
                    )
                    print(f"[EssayScoring] WARNING: {self._load_error}")
                    return

            self._model = joblib.load(model_path)
            self._scaler = joblib.load(scaler_path)
            self._loaded = True
            print("[EssayScoring] Model and Scaler loaded successfully!")

        except Exception as e:
            self._load_error = str(e)
            print(f"[EssayScoring] ERROR saat memuat model: {e}")

    @property
    def is_available(self) -> bool:
        return self._loaded

    def score_essay(self, essay: str, reference: str) -> dict:
        """
        Hitung skor esai siswa (0-100) berdasarkan kunci jawaban.
        """
        if not self._loaded:
            raise ValueError(f"Model Penilaian Esai tidak tersedia: {self._load_error}")

        essay_clean = essay.strip()
        ref_clean = reference.strip()

        # 1. is_blank
        is_blank = 1.0 if len(essay_clean) == 0 else 0.0

        # Ekstraksi kata
        words_essay = re.findall(r'\b\w+\b', essay_clean.lower())
        words_ref = re.findall(r'\b\w+\b', ref_clean.lower())

        len_essay = len(words_essay)
        len_ref = len(words_ref)

        # 2. length_ratio
        length_ratio = len_essay / max(len_ref, 1)

        # 3. is_short (jika lebih pendek dari 50% kunci jawaban atau di bawah 10 kata)
        is_short = 1.0 if (length_ratio < 0.5 or len_essay < 10) and not is_blank else 0.0

        # 4. sequence_similarity (difflib matcher ratio)
        import difflib
        sequence_sim = difflib.SequenceMatcher(None, essay_clean.lower(), ref_clean.lower()).ratio()

        # 5. keyword_overlap (mengabaikan stopwords umum bahasa Indonesia)
        stopwords_id = {
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'ini', 'itu', 'adalah',
            'yaitu', 'yakni', 'sebagai', 'oleh', 'pada', 'atau', 'juga', 'dalam', 'akan'
        }
        set_essay = set(words_essay) - stopwords_id
        set_ref = set(words_ref) - stopwords_id

        if not set_ref:
            keyword_overlap = 0.0
        else:
            keyword_overlap = len(set_essay & set_ref) / len(set_ref)

        # 6. cosine_similarity (SBERT)
        if is_blank:
            cos_sim = 0.0
        else:
            try:
                embeddings = self.sbert_model.encode([essay_clean, ref_clean], convert_to_numpy=True)
                cos_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            except Exception as e:
                print(f"[EssayScoring] Error saat menghitung cosine similarity: {e}")
                cos_sim = 0.0

        # Urutan fitur sesuai meta.json:
        # [cosine_similarity, keyword_overlap, sequence_similarity, length_ratio, is_short, is_blank]
        features = [cos_sim, keyword_overlap, sequence_sim, length_ratio, is_short, is_blank]

        # Skala fitur menggunakan StandardScaler
        X = np.array([features], dtype=np.float32)
        X_scaled = self._scaler.transform(X)

        # Prediksi skor kontinu
        raw_score = float(self._model.predict(X_scaled)[0])

        # Clip skor ke [0.0, 1.0] dan konversi ke persen (0-100)
        final_score = round(max(0.0, min(1.0, raw_score)) * 100, 2)

        return {
            "score": final_score,
            "features": {
                "cosine_similarity": round(cos_sim * 100, 2),
                "keyword_overlap": round(keyword_overlap * 100, 2),
                "sequence_similarity": round(sequence_sim * 100, 2),
                "length_ratio": round(length_ratio * 100, 2),
                "is_short": bool(is_short),
                "is_blank": bool(is_blank),
                "word_count_student": len_essay,
                "word_count_reference": len_ref
            }
        }

