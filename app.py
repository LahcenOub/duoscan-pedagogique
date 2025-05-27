import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px # Pour Plotly Express
import re

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="DuoScan Pédagogique",
    page_icon="🦉",
    layout="wide"
)

# --- Fonctions Utilitaires (inchangées) ---
def nettoyer_texte(texte):
    texte = texte.lower()
    return texte

def analyser_sentiment(texte):
    blob = TextBlob(texte)
    polarite = blob.sentiment.polarity
    if polarite > 0.1:
        return "Positif 👍"
    elif polarite < -0.1:
        return "Négatif 👎"
    else:
        return "Neutre 😐"

def generer_nuage_mots(textes, titre_section):
    if not textes:
        st.info(f"Pas de données suffisantes pour générer le nuage de mots {titre_section.lower()}.")
        return

    texte_concatene = " ".join(nettoyer_texte(txt) for txt in textes)
    if not texte_concatene.strip():
        st.info(f"Pas de mots significatifs pour le nuage {titre_section.lower()}.")
        return

    try:
        wordcloud = WordCloud(width=800, height=400, background_color='rgba(255, 255, 255, 0)',
                              collocations=False, prefer_horizontal=0.9).generate(texte_concatene)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors de la génération du nuage de mots '{titre_section}': {e}")

# --- Barre Latérale (Sidebar) ---
with st.sidebar:
    # st.image("https://i.imgur.com/5RfOsTQ.png", width=100) # Ligne de l'image du logo supprimée
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
        "pour identifier des pistes d'amélioration "
        "des expériences d'apprentissage numérique."
    )
    st.markdown("### Guide Rapide")
    st.markdown("""
    1.  Choisissez la méthode d'entrée.
    2.  Fournissez les avis Duolingo.
    3.  Cliquez sur "🚀 Analyser les Avis".
    4.  Explorez les insights !
    """)
    st.divider()
    st.caption(f"© {pd.Timestamp.now().year} EL FILALI Mohamed")


# --- Interface Utilisateur Principale ---
st.title("🦉 DuoScan Pédagogique")
st.subheader("Analyse des Avis Utilisateurs de Duolingo pour l'Optimisation des Expériences d'Apprentissage")
st.markdown("Bienvenue ! Cet outil vous aide à sonder le ressenti exprimé dans les avis Duolingo.")
st.divider()

# --- Section d'Entrée des Données ---
st.header("📥 1. Soumettre les Avis Duolingo")
input_method = st.radio(
    "Comment souhaitez-vous fournir les avis Duolingo ?",
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
        placeholder="Exemple : Duolingo m'a vraiment aidé à apprendre l'espagnol, c'est ludique !\nParfois, les notifications sont un peu trop insistantes."
    )
    if avis_texte_area:
        avis_entres_bruts = [avis.strip() for avis in avis_texte_area.split('\n') if avis.strip()]

elif input_method == "Télécharger un fichier (.txt ou .csv)":
    uploaded_file = st.file_uploader(
        "Sélectionnez un fichier .txt (un avis Duolingo par ligne) ou .csv (avec une colonne 'avis' ou une seule colonne)",
        type=["txt", "csv"],
        key="file_uploader_input"
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.txt'):
                avis_entres_bruts = [line.decode('utf-8').strip() for line in uploaded_file if line.decode('utf-8').strip()]
            elif uploaded_file.name.endswith('.csv'):
                df_avis = pd.read_csv(uploaded_file)
                if 'avis' in df_avis.columns:
                    avis_entres_bruts = [str(avis).strip() for avis in df_avis['avis'].dropna().tolist() if str(avis).strip()]
                elif len(df_avis.columns) == 1:
                    avis_entres_bruts = [str(avis).strip() for avis in df_avis.iloc[:, 0].dropna().tolist() if str(avis).strip()]
                else:
                    st.error("Fichier CSV : veuillez fournir une colonne nommée 'avis' ou un fichier avec une seule colonne.")
        except Exception as e:
            st.error(f"⚠️ Erreur lors de la lecture du fichier : {e}")
            st.stop()

# --- Bouton d'Analyse ---
st.divider()
if st.button("🚀 Analyser les Avis Duolingo", type="primary", use_container_width=True, key="analyze_button"):
    if not avis_entres_bruts:
        st.warning("⚠️ Veuillez fournir des avis Duolingo avant de lancer l'analyse.")
        st.stop()

    st.header("📊 2. Analyse des Sentiments (Avis Duolingo)")
    with st.spinner("🦉 Analyse en cours... Un instant, je consulte les chouettes !"):
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

        # --- Affichage des Indicateurs Clés ---
        st.subheader("📈 Vue d'Ensemble des Retours sur Duolingo")
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
        # --- Détail par Avis ---
        with st.expander("🔍 Explorer chaque avis Duolingo individuellement", expanded=False):
            if avis_analyses:
                df_details = pd.DataFrame(avis_analyses)
                df_details = df_details[['id', 'sentiment_detecte', 'texte_original']]
                df_details.rename(columns={'id':'ID', 'sentiment_detecte':'Sentiment Détecté', 'texte_original':'Texte de l\'Avis Duolingo'}, inplace=True)
                st.dataframe(df_details, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun avis à afficher en détail.")

        st.divider()
        # --- Nuages de Mots ---
        st.subheader("☁️ Termes Fréquents dans les Avis Duolingo")
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

        # --- Section d'Interprétation ---
        st.header("💡 3. Pistes pour le Design de l'Expérience d'Apprentissage sur Duolingo")
        st.markdown("""
            À partir des sentiments et des termes identifiés dans les avis Duolingo :
            * Quels sont les **points forts** perçus par les utilisateurs de Duolingo ? (ex: gamification, variété des exercices, facilité d'accès...)
            * Quelles **frustrations ou difficultés** émergent ? (ex: système de vies, publicités, progression, type d'exercices...)
            * Comment ces retours peuvent-ils éclairer l'application des **principes de design UX pour l'apprentissage** à Duolingo ?
            * Quelles **recommandations spécifiques** pourriez-vous formuler pour améliorer l'expérience pédagogique sur Duolingo, en vous basant sur ces données ?
        """)
        st.text_area(
            "Rédigez ici votre analyse et vos recommandations pour M. AMMAR :",
            height=200,
            key="interpretation_area_detailed",
            placeholder="Exemple : L'analyse suggère que l'aspect ludique de Duolingo est un levier majeur d'engagement. Cependant, la récurrence de termes négatifs liés aux 'publicités' ou aux 'vies' pourrait indiquer des points de friction impactant la persévérance..."
        )
else:
    st.info("ℹ️ Prêt à scanner les avis Duolingo ? Fournissez des données et cliquez sur 'Analyser'.")