import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import re
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
import nltk
import logging
import chardet

# Download NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="DuoScan Pédagogique",
    page_icon="🦉",
    layout="wide"
)

# --- Utility Functions ---
def nettoyer_texte(texte):
    """Clean text by converting to lowercase, removing punctuation, numbers, and stopwords."""
    if not texte or not isinstance(texte, str):
        return ""
    texte = texte.lower()
    texte = re.sub(r'[^\w\s]', '', texte)  # Remove punctuation
    texte = re.sub(r'\d+', '', texte)  # Remove numbers
    stop_words = set(stopwords.words('french')) | set(['duolingo', 'lapplication', 'cest'])
    texte = " ".join(word for word in texte.split() if word not in stop_words)
    return texte

def translate_to_english(texte):
    """Translate French text to English for VADER analysis."""
    try:
        return GoogleTranslator(source='fr', target='en').translate(texte)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return texte  # Fallback to original text

analyzer = SentimentIntensityAnalyzer()

def analyser_sentiment(texte):
    """Analyze sentiment using VADER on translated text with French negative term detection."""
    if not texte or not texte.strip():
        return "Neutre 😐"
    
    # Lexique français pour termes fortement négatifs
    negative_terms = {'nul', 'nulle', 'horrible', 'affreux', 'détestable', 'pénible', 'agaçant', 'ennuyeux'}
    texte_lower = texte.lower()
    
    # Vérifier si le texte contient des termes négatifs explicites
    if any(term in texte_lower.split() for term in negative_terms):
        return "Négatif 👎"
    
    # Traduire en anglais pour VADER
    texte_en = translate_to_english(texte)
    vs = analyzer.polarity_scores(texte_en)
    compound_score = vs['compound']
    
    # Seuils ajustés pour plus de sensibilité
    if compound_score >= 0.05:
        return "Positif 👍"
    elif compound_score <= -0.05:
        return "Négatif 👎"
    else:
        return "Neutre 😐"

def generer_nuage_mots(textes, titre_section):
    """Generate word cloud from list of texts."""
    if not textes:
        st.info(f"Pas de données suffisantes pour générer le nuage de mots {titre_section.lower()}.")
        return

    texte_concatene = " ".join(nettoyer_texte(txt) for txt in textes if txt)
    if not texte_concatene.strip():
        st.info(f"Pas de mots significatifs pour le nuage {titre_section.lower()}.")
        return

    try:
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='rgba(255, 255, 255, 0)',
            collocations=False,
            prefer_horizontal=0.9,
            min_font_size=10
        ).generate(texte_concatene)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors de la génération du nuage de mots '{titre_section}': {e}")

def detect_encoding(file):
    """Detect file encoding."""
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding']

# --- Sidebar ---
with st.sidebar:
    st.image("https://i.imgur.com/LF3KIQa.jpeg", width=100, caption="EL FILALI MOHAMED")
    st.markdown("# DuoScan Pédagogique 🦉")
    st.divider()
    st.markdown("**Master :** Ingénierie Technopédagogique et Innovation")
    st.markdown("**Module :** Design de l'Expérience d'Apprentissage")
    st.markdown("**Professeur :** M. Adil AMMAR")
    st.markdown("**Réalisé par :** EL FILALI Mohamed")
    st.divider()
    st.markdown("### Objectif de l'outil:")
    st.info(
        "Analyser le sentiment des retours utilisateurs de Duolingo "
        "pour identifier des pistes d'amélioration des expériences d'apprentissage."
    )
    st.markdown("### Guide Rapide")
    st.markdown("""
    1. Choisissez la méthode d'entrée.
    2. Fournissez les avis Duolingo.
    3. Cliquez sur "🚀 Analyser les Avis".
    4. Explorez les insights !
    """)
    st.divider()
    st.caption(f"© {pd.Timestamp.now().year} - ITPI")

# --- Main Interface ---
st.title("🦉 DuoScan Pédagogique")
st.subheader("Analyse des Avis Utilisateurs de Duolingo pour l'Optimisation des Expériences d'Apprentissage")
st.markdown("Cet outil analyse les sentiments exprimés dans les avis Duolingo.")
st.divider()

# --- Data Input Section ---
st.header("📥 1. Soumettre les Avis Duolingo")
input_method = st.radio(
    "Comment souhaitez-vous fournir les avis ?",
    ("Coller le texte directement", "Télécharger un fichier (.txt ou .csv)"),
    key="input_method_choice",
    horizontal=True
)

avis_entres_bruts = []

if input_method == "Coller le texte directement":
    avis_texte_area = st.text_area(
        "Collez ici les avis Duolingo (un par ligne) :",
        height=150,
        key="text_area_input",
        placeholder="Exemple : Duolingo m'a vraiment aidé à apprendre l'espagnol, c'est ludique !\nLes notifications sont trop insistantes."
    )
    if avis_texte_area:
        avis_entres_bruts = [avis.strip() for avis in avis_texte_area.split('\n') if avis.strip()]

elif input_method == "Télécharger un fichier (.txt ou .csv)":
    uploaded_file = st.file_uploader(
        "Sélectionnez un fichier .txt (un avis par ligne) ou .csv (colonne 'avis' ou unique colonne)",
        type=["txt", "csv"],
        key="file_uploader_input"
    )
    if uploaded_file is not None:
        try:
            encoding = detect_encoding(uploaded_file)
            if uploaded_file.name.endswith('.txt'):
                avis_entres_bruts = [line.decode(encoding).strip() for line in uploaded_file if line.decode(encoding).strip()]
            elif uploaded_file.name.endswith('.csv'):
                df_avis = pd.read_csv(uploaded_file, encoding=encoding)
                if 'avis' in df_avis.columns:
                    avis_entres_bruts = [str(avis).strip() for avis in df_avis['avis'].dropna().tolist() if str(avis).strip()]
                elif len(df_avis.columns) == 1:
                    avis_entres_bruts = [str(avis).strip() for avis in df_avis.iloc[:, 0].dropna().tolist() if str(avis).strip()]
                else:
                    st.error("Fichier CSV : utilisez une colonne nommée 'avis' ou une seule colonne.")
                    st.stop()
        except Exception as e:
            st.error(f"⚠️ Erreur lors de la lecture du fichier : {e}. Essayez un fichier encodé en UTF-8.")
            st.stop()

# --- Analysis Button ---
st.divider()
if st.button("🚀 Analyser les Avis Duolingo", type="primary", use_container_width=True, key="analyze_button"):
    if not avis_entres_bruts:
        st.warning("⚠️ Veuillez fournir des avis Duolingo avant de lancer l'analyse.")
        st.stop()

    # Limit input size
    if len(avis_entres_bruts) > 1000:
        st.warning("⚠️ Trop d'avis fournis (limite : 1000). Veuillez réduire la taille du dataset.")
        st.stop()

    st.header("📊 2. Analyse des Sentiments")
    with st.spinner("🦉 Analyse en cours..."):
        avis_analyses = []
        sentiments_counts = {"Positif 👍": 0, "Négatif 👎": 0, "Neutre 😐": 0}
        avis_positifs_textes = []
        avis_negatifs_textes = []

        for i, avis_texte in enumerate(avis_entres_bruts):
            sentiment = analyser_sentiment(avis_texte)
            avis_analyses.append({"id": i + 1, "texte_original": avis_texte, "sentiment_detecte": sentiment})
            sentiments_counts[sentiment] += 1
            if sentiment == "Positif 👍":
                avis_positifs_textes.append(avis_texte)
            elif sentiment == "Négatif 👎":
                avis_negatifs_textes.append(avis_texte)

        st.toast('Analyse terminée ! 🎉', icon='✅')

        # --- Key Metrics ---
        st.subheader("📈 Vue d'Ensemble")
        cols_metriques = st.columns(4)
        cols_metriques[0].metric("Total Avis Analysés", len(avis_analyses))
        cols_metriques[1].metric("Avis Positifs 👍", sentiments_counts["Positif 👍"])
        cols_metriques[2].metric("Avis Négatifs 👎", sentiments_counts["Négatif 👎"])
        cols_metriques[3].metric("Avis Neutres 😐", sentiments_counts["Neutre 😐"])

        if sum(sentiments_counts.values()) > 0:
            df_sentiments = pd.DataFrame(list(sentiments_counts.items()), columns=['Sentiment', 'Nombre'])
            sentiment_color_map = {
                "Positif 👍": "#28a745",
                "Négatif 👎": "#dc3545",
                "Neutre 😐": "#6c757d"
            }
            fig = px.bar(
                df_sentiments,
                x='Sentiment',
                y='Nombre',
                color='Sentiment',
                color_discrete_map=sentiment_color_map,
                labels={'Nombre': "Nombre d'Avis", 'Sentiment': 'Catégorie de Sentiment'},
                text_auto=True
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title=None,
                yaxis_title="Nombre d'Avis",
                font=dict(family="sans-serif", size=12),
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée à afficher dans le graphique.")

        st.divider()
        # --- Detailed Analysis ---
        with st.expander("🔍 Explorer chaque avis", expanded=False):
            if avis_analyses:
                df_details = pd.DataFrame(avis_analyses)
                df_details = df_details[['id', 'sentiment_detecte', 'texte_original']]
                df_details.rename(columns={'id':'ID', 'sentiment_detecte':'Sentiment Détecté', 'texte_original':'Texte de l\'Avis'}, inplace=True)
                st.dataframe(df_details, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun avis à afficher en détail.")

        st.divider()
        # --- Word Clouds ---
        st.subheader("☁️ Termes Fréquents")
        show_wordclouds = st.checkbox("Afficher les nuages de mots", value=True, key="show_wc")
        if show_wordclouds:
            if not avis_positifs_textes and not avis_negatifs_textes:
                st.info("Pas assez de données textuelles pour générer les nuages de mots.")
            else:
                tab_positif, tab_negatif = st.tabs(["Termes Positifs", "Termes Négatifs"])
                with tab_positif:
                    generer_nuage_mots(avis_positifs_textes, "issus des Avis Positifs")
                with tab_negatif:
                    generer_nuage_mots(avis_negatifs_textes, "issus des Avis Négatifs")
        st.divider()

        # --- Interpretation Section ---
        st.header("💡 3. Pistes pour l'Expérience d'Apprentissage")
        positive_ratio = sentiments_counts["Positif 👍"] / len(avis_analyses) if avis_analyses else 0
        negative_ratio = sentiments_counts["Négatif 👎"] / len(avis_analyses) if avis_analyses else 0
        insights = []
        if positive_ratio > 0.5:
            insights.append("✅ Les utilisateurs apprécient majoritairement Duolingo, probablement grâce à son aspect ludique.")
        if negative_ratio > 0.3:
            insights.append("⚠️ De nombreuses frustrations émergent, potentiellement liées à des publicités ou restrictions.")
        if avis_positifs_textes:
            insights.append("💡 Points forts : examinez les termes dans les avis positifs pour identifier les fonctionnalités plébiscitées.")
        if avis_negatifs_textes:
            insights.append("🔍 Points à améliorer : les termes négatifs indiquent des irritants UX à corriger.")

        if insights:
            st.markdown("### Insights Automatiques")
            for insight in insights:
                st.markdown(f"- {insight}")

        st.markdown("""
            À partir des sentiments et termes identifiés :
            * Quels sont les **points forts** perçus par les utilisateurs ?
            * Quelles **frustrations ou difficultés** émergent ?
            * Comment ces retours éclairent-ils les **principes de design UX** ?
            * Quelles **recommandations** pour améliorer l'expérience pédagogique ?
        """)
        st.text_area(
            "Rédigez votre analyse et recommandations pour M. AMMAR :",
            height=200,
            key="interpretation_area_detailed",
            placeholder="Exemple : Les avis positifs soulignent la gamification. Les termes négatifs comme 'publicités' suggèrent des interruptions..."
        )
else:
    st.info("ℹ️ Prêt à scanner les avis Duolingo ? Fournissez des données et cliquez sur 'Analyser'.")