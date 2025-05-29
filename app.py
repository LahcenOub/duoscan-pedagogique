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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import io

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
    page_title="DuoScan Pédagogique", # MODIFICATION: enlevé "Avancé" ici aussi pour cohérence
    page_icon="🦉",
    layout="wide"
)

# --- Utility Functions ---
def nettoyer_texte(texte):
    if not texte or not isinstance(texte, str):
        return ""
    texte = texte.lower()
    texte = re.sub(r'[^\w\s]', '', texte)
    texte = re.sub(r'\d+', '', texte)
    stop_words = set(stopwords.words('french')) | set([
        'duolingo', 'lapplication', 'cest', 'app', 'jai', 'suis', 'cest', 'était',
        'plus', 'très', 'fait', 'faire', 'fais', 'faites',
        'bien', 'bon', 'bonne', 'super', 'génial', 'excellent',
        'mauvais', 'nul', 'horrible',
        'toujours', 'merci', 'vraiment', 'svp', 'sil vous plaît',
        'ça', 'cela', 'ceci', 'ici',
        'être', 'avoir', 'vouloir', 'pouvoir', 'devoir',
        'depuis', 'peut', 'aussi', 'comme', 'idem',
        'pourquoi', 'quand', 'comment', 'donc', 'alors',
        'tout', 'tous', 'toute', 'toutes', 'rien', 'personne',
        'fois', 'jour', 'jours', 'semaine', 'mois', 'année',
        'compte', 'problème', 'question', 'chose', 'truc', 'machin',
        'utilisateur', 'interface', 'utilisation', 'version', 'option', 'fonctionnalité',
        'début', 'fin', 'milieu', 'partie', 'niveau', 'leçon', 'exercice',
        'dit', 'dire', 'parler', 'voir', 'mettre', 'prendre', 'donner'
    ])
    texte = " ".join(word for word in texte.split() if word not in stop_words and len(word) > 1)
    return texte

def translate_to_english(texte):
    if not texte or not isinstance(texte, str) or not texte.strip():
        return ""
    try:
        if len(texte) > 4800:
            logger.warning(f"Texte trop long ({len(texte)} char.) pour traduction auto, analyse VADER sur original si lexique FR pertinent.")
            return texte
        return GoogleTranslator(source='fr', target='en').translate(texte)
    except Exception as e:
        logger.error(f"Translation error for text snippet '{texte[:50]}...': {e}")
        return texte

analyzer = SentimentIntensityAnalyzer()

def analyser_sentiment(texte_original):
    if not texte_original or not isinstance(texte_original, str) or not texte_original.strip():
        return "Neutre 😐"
    texte_original_lower = texte_original.lower()
    negative_terms_fr = {
        'abusé', 'agaçant', 'agaçante', 'arnaque', 'arrête pas', 'arrêter de',
        'attendre longtemps', 'aucun progrès', 'aucun sens', 'aucune logique',
        'affreux', 'affreuse', 'bug', 'buggé', 'beug', 'beugue',
        'casse les pieds', 'cétait mieux avant', 'chiant', 'complexe', 'compliqué',
        'débile', 'déçu', 'décevante', 'décroire', 'décourageant', 'dégoûté',
        'demande toujours', 'dépensé pour rien', 'désagréable', 'détestable', 'difficile',
        'dommage', 'ennuyeux', 'ennuyeuse', 'erreur', 'erreurs fréquentes',
        'fatigant', 'faux', 'forcé de', 'frustrant', 'frustrante',
        'gâche', 'gênant', 'horrible', 'honteux',
        'impossible', 'inaccessible', 'inacceptable', 'incompréhensible', 'incohérent',
        'incorrect', 'infesté de pub', 'injouable', 'insupportable', 'instable', 'insuffisant',
        'inutile', 'lent', 'lenteur', 'limité',
        'mal fait', 'mal fichu', 'malheureusement', 'manque de', 'marche pas', 'mauvais', 'mauvaise',
        'moche', 'mensonge', 'médiocre',
        'ne fonctionne plus', 'ne marche plus', 'ne répond pas', 'négatif',
        'nul', 'nulle', 'nuls', 'obligé de', 'obsolète',
        'pas assez', 'pas clair', 'pas du tout', 'pas efficace', 'pas terrible', 'pas top',
        'pénible', 'perdu', 'perte de temps', 'pire', 'plante', 'plein de bugs', 'popups',
        'pourri', 'pourrie', 'problème', 'problèmes',
        'ralentit', 'rame', 'regrettable', 'régressé', 'ridicule',
        'sans intérêt', 'saturation', 'saturé de pub', 'souci', 'stupide',
        'trop de pub', 'trop cher', 'trop difficile', 'trop simple', 'triste',
        'usant', 'usine à gaz', 'vide', 'vieillot', 'violent', 'zero'
    }
    if any(term in texte_original_lower for term in negative_terms_fr):
        if not (("n'est pas" in texte_original_lower or "ne pas" in texte_original_lower or "pas si" in texte_original_lower) and \
                any(term in texte_original_lower for term in ['nul', 'mauvais', 'décevant', 'horrible'])):
            return "Négatif 👎"

    texte_en = translate_to_english(texte_original)
    if texte_en == texte_original or not texte_en:
        pass

    vs = analyzer.polarity_scores(texte_en)
    compound_score = vs['compound']

    if compound_score >= 0.05:
        return "Positif 👍"
    elif compound_score <= -0.04:
        return "Négatif 👎"
    else:
        subtle_negative_hints_fr = {'un peu déçu', 'pas vraiment top', 'pourrait être mieux', 'bof'}
        if any(hint in texte_original_lower for hint in subtle_negative_hints_fr):
            return "Négatif 👎"
        return "Neutre 😐"

def generer_nuage_mots(textes, titre_section):
    if not textes:
        st.info(f"Pas de données suffisantes pour générer le nuage de mots {titre_section.lower()}.")
        return
    texte_concatene = " ".join(nettoyer_texte(txt) for txt in textes if txt and isinstance(txt, str))
    if not texte_concatene.strip():
        st.info(f"Pas de mots significatifs (après nettoyage) pour le nuage {titre_section.lower()}.")
        return
    try:
        wordcloud = WordCloud(
            width=800, height=400, background_color='rgba(255, 255, 255, 0)',
            collocations=True, prefer_horizontal=0.9, min_font_size=10,
        ).generate(texte_concatene)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    except ValueError as ve:
         st.warning(f"Impossible de générer le nuage de mots '{titre_section}' : {ve}. ")
    except Exception as e:
        st.error(f"Erreur lors de la génération du nuage de mots '{titre_section}': {e}")

def detect_encoding(file):
    try:
        raw_data = file.read(100000)
        result = chardet.detect(raw_data)
        file.seek(0)
        if result['encoding'] and result['confidence'] > 0.7:
            return result['encoding']
        logger.info(f"Chardet low confidence ({result['confidence']}) for detected encoding '{result['encoding']}'. Defaulting to utf-8.")
        return 'utf-8'
    except Exception as e:
        logger.error(f"Erreur lors de la détection de l’encodage : {e}")
        return 'utf-8'

def detect_topics(textes, num_topics=5, n_top_words=7):
    valid_texts = [txt for txt in textes if txt and isinstance(txt, str)]
    if not valid_texts or len(valid_texts) < 2: return []
    try:
        textes_nettoyes = [nettoyer_texte(txt) for txt in valid_texts]
        textes_nettoyes = [txt for txt in textes_nettoyes if txt and len(txt.split()) > 3]
        if len(textes_nettoyes) < num_topics:
            num_topics = max(1, len(textes_nettoyes))
        if len(textes_nettoyes) < 2:
            logger.info("Pas assez de documents valides pour la détection de thèmes après nettoyage.")
            return []
        vectorizer = CountVectorizer(max_df=0.90, min_df=3, stop_words='english', ngram_range=(1,2), max_features=1000)
        doc_term_matrix = vectorizer.fit_transform(textes_nettoyes)
        if doc_term_matrix.shape[0] < num_topics or doc_term_matrix.shape[1] == 0:
            logger.warning(f"Matrice document-terme ({doc_term_matrix.shape}) insuffisante pour LDA avec {num_topics} thèmes.")
            return []
        if doc_term_matrix.shape[1] < n_top_words :
             n_top_words = doc_term_matrix.shape[1]
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, learning_method='online', batch_size=128)
        lda.fit(doc_term_matrix)
        feature_names = vectorizer.get_feature_names_out()
        topics_summary = []
        for topic_idx, topic_weights in enumerate(lda.components_):
            top_words_indices = topic_weights.argsort()[:-min(n_top_words + 1, len(feature_names) + 1):-1]
            top_words = [feature_names[i] for i in top_words_indices]
            topics_summary.append(f"Thème {topic_idx + 1}: {', '.join(top_words)}")
        return topics_summary
    except Exception as e:
        logger.error(f"Erreur lors de la détection des thèmes : {e}")
        return []

def analyze_lxd_aspects(textes, sentiments):
    lxd_aspects = {
        "Interface Utilisateur (UI)": ["navigation", "intuitive", "clair", "visuel", "menu", "bouton", "ergonomie", "design pattern"],
        "Personnalisation & Adaptabilité": ["personnalisé", "adapté", "objectif", "rythme", "niveau", "adaptive", "sur mesure", "parcours"],
        "Esthétique & Engagement Émotionnel": ["design", "couleur", "ludique", "attrayant", "hibou", "mascotte", "amusant", "plaisant", "joli", "graphisme"],
        "Gamification & Motivation": ["points", "xp", "gemmes", "lingots", "badges", "séries", "streak", "classement", "ligue", "défi", "récompense", "vies", "coeurs", "challenge"],
        "Contenu Pédagogique & Exercices": ["contenu", "cours", "vocabulaire", "grammaire", "phrases", "traduction", "répétitif", "variété des exercices", "qualité du contenu"],
        "Feedback & Progression": ["feedback", "correction", "explication", "aide", "progression", "progrès", "suivi", "encouragement"],
        "Accessibilité & Flexibilité": ["mobile", "gratuit", "freemium", "rapide", "courtes", "temps", "hors ligne", "accessibilité"],
        "Publicité & Modèle Économique": ["pub", "publicité", "premium", "abonnement", "payant", "gratuit vs payant", "coût"]
    }
    aspect_sentiments = {aspect: {"Positif": 0, "Négatif": 0, "Neutre": 0, "Mentions":0} for aspect in lxd_aspects}
    for texte, sentiment in zip(textes, sentiments):
        if not texte or not isinstance(texte, str): continue
        texte_lower_cleaned = nettoyer_texte(texte).lower()
        texte_original_lower = texte.lower()
        sentiment_key = sentiment.split(" ")[0]
        for aspect, keywords in lxd_aspects.items():
            if any(keyword in texte_lower_cleaned or keyword in texte_original_lower for keyword in keywords):
                if sentiment_key in aspect_sentiments[aspect]:
                    aspect_sentiments[aspect][sentiment_key] += 1
                aspect_sentiments[aspect]["Mentions"] +=1
    return aspect_sentiments

def generate_theory_recommendations(avis_negatifs_textes, avis_positifs_textes, lxd_results):
    recommendations = []
    texte_neg_concatene = " ".join(nettoyer_texte(txt).lower() for txt in avis_negatifs_textes if txt)
    negative_themes_to_theories = {
        "vies": ("Théorie du Flow & Auto-Détermination", "Le système de vies/cœurs est un point de friction majeur. Il interrompt le flow (Csikszentmihalyi) et peut nuire au sentiment de compétence (SDT). Proposer un mode 'entraînement libre' sans vies ou des options pour regagner des vies plus facilement est crucial."),
        "bloqué": ("Théorie du Flow", "Les blocages dus aux erreurs ou au système de vies interrompent le flow. Envisager des aides contextuelles plus poussées, des 'jokers', ou des mini-révisions ciblées pour surmonter les obstacles et maintenir l'engagement."),
        "publicités": ("Théorie de l’Auto-Détermination & Flow", "Les publicités fréquentes sont perçues comme une nuisance majeure, nuisant à l'autonomie et interrompant le flow. Réduire leur fréquence/intrusivité ou offrir des contreparties claires (ex: gagner des vies) pour les visionnages volontaires."),
        "répétitif": ("Critique du Behaviorisme & Théorie Cognitive de l'Apprentissage Multimédia (Mayer)", "La répétition excessive engendre monotonie et désengagement. Introduire plus de variété dans les types d'exercices (cf. LXD aspect 'Contenu Pédagogique') et contextualiser davantage l'apprentissage (scénarios, dialogues) pour favoriser un encodage profond et réduire la charge cognitive extrinsèque."),
        "mécanique": ("Critique du Behaviorisme", "Un apprentissage perçu comme trop mécanique limite le transfert des compétences. Intégrer des tâches qui sollicitent la réflexion critique ou la créativité (composition libre, résolution de problèmes en contexte)."),
        "manque contexte": ("Critique du Behaviorisme & Constructivisme", "Le manque de contexte nuit à la compréhension et à l'application réelle. Renforcer l'utilisation de scénarios authentiques, d'histoires interactives plus élaborées et de mises en situation pratiques."),
        "pas explication": ("Behaviorisme (Feedback) & Cognitivisme", "Un feedback insuffisant ou peu clair est un frein majeur. Fournir des explications grammaticales et lexicales détaillées, accessibles à la demande, et multimodales (texte, audio, exemples concrets). Le feedback doit être constructif et guider l'apprenant."),
        "grammaire difficile": ("Cognitivisme & Feedback", "Les difficultés grammaticales nécessitent un soutien pédagogique accru. Proposer des mini-leçons de grammaire ciblées, des tableaux récapitulatifs, des exemples clairs et des exercices spécifiques pour chaque point complexe. Le séquençage doit être progressif."),
        "pression": ("Théorie de l’Auto-Détermination (Compétence & Autonomie)", "La pression des streaks ou des classements peut être anxiogène. Offrir des options pour désactiver/masquer ces éléments compétitifs ou proposer des modes d'apprentissage axés sur la maîtrise personnelle plutôt que la compétition."),
        "classement": ("Théorie de l’Auto-Détermination (Compétence & Relation)", "Les classements peuvent démotiver ceux qui ne sont pas en tête. Envisager des formes de 'compétition saine' (ex: défis de groupe contre un objectif commun) ou des comparaisons de progrès personnels pour renforcer la relatedness positivement."),
        "notifications": ("Théorie de l’Auto-Détermination (Autonomie)", "Les notifications perçues comme trop insistantes ou non pertinentes peuvent nuire au sentiment d'autonomie. Permettre une personnalisation fine des rappels et s'assurer qu'ils sont perçus comme un soutien et non un contrôle."),
        "interaction": ("Constructivisme Social (Vygotsky)", "Un manque d'interaction sociale est souvent relevé. Explorer des fonctionnalités d'apprentissage collaboratif modéré : forums de discussion par leçon, correction par les pairs (simple), défis collaboratifs."),
        "parler": ("Constructivisme & Apprentissage Actif", "La difficulté à transférer les acquis à l'oral est une préoccupation majeure. Intégrer significativement plus d'exercices de production orale (reconnaissance vocale améliorée, scénarios de dialogue interactif, feedback sur la prononciation)."),
        "écrire": ("Constructivisme & Apprentissage Actif", "Le manque de pratique en production écrite est une limite. Proposer des exercices de rédaction guidée, de résumé, ou de petites descriptions en lien avec les thèmes étudiés."),
        "trop cher": ("Accessibilité & Modèle Freemium (LXD)", "Le coût de la version premium est un frein important. Réévaluer la proposition de valeur du mode gratuit versus payant. Envisager des options d'abonnement plus flexibles ou des fonctionnalités premium accessibles ponctuellement via des 'crédits' gagnés."),
        "lent": ("Performance & UX (LXD)", "Les lenteurs et les bugs dégradent fortement l'expérience utilisateur. Prioriser l'optimisation des performances de l'application et la correction des bugs signalés."),
        "bug": ("Fiabilité & UX (LXD)", "Les bugs fréquents sapent la confiance et la motivation. Mettre en place un processus rigoureux de tests et de résolution rapide des bugs, et communiquer sur les correctifs apportés."),
    }
    for aspect, counts in lxd_results.items():
        if counts["Mentions"] > 0:
            negative_ratio_for_aspect = counts["Négatif"] / counts["Mentions"]
            if negative_ratio_for_aspect > 0.4 and counts["Négatif"] > 3 :
                action = f"L'aspect LXD '{aspect}' reçoit une proportion significative d'avis négatifs ({counts['Négatif']}/{counts['Mentions']}). Il est crucial d'investiguer les causes spécifiques (voir avis détaillés) et d'envisager des améliorations ciblées."
                if aspect == "Publicité & Modèle Économique" and "publicités" not in texte_neg_concatene:
                     recommendations.append(f"🔬 **LXD Critique & Auto-Détermination**: {action}")
                elif aspect == "Contenu Pédagogique & Exercices" and "répétitif" not in texte_neg_concatene:
                     recommendations.append(f"🔬 **LXD Critique & Cognitivisme**: {action}")
                elif aspect not in ["Publicité & Modèle Économique", "Contenu Pédagogique & Exercices"]:
                    recommendations.append(f"🔬 **LXD Critique**: {action}")

    triggered_recommendations_texts = set(rec.split(": ", 1)[1] for rec in recommendations)
    if texte_neg_concatene:
        for keyword, (theory, recommendation_text) in negative_themes_to_theories.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', texte_neg_concatene):
                if recommendation_text not in triggered_recommendations_texts:
                    recommendations.append(f"🔬 **{theory}**: {recommendation_text}")
                    triggered_recommendations_texts.add(recommendation_text)

    if not recommendations and avis_negatifs_textes:
        recommendations.append("🔍 **Point d'Attention Général**: Des avis négatifs ont été détectés. Il est conseillé de les examiner manuellement pour identifier des problèmes spécifiques non couverts par les thèmes automatisés. L'amélioration continue du feedback, la variété des exercices et la gestion de la frustration sont des pistes universelles.")
    elif not avis_negatifs_textes and avis_positifs_textes:
        recommendations.append("🎉 **Excellente Réception Générale**: Les retours sont majoritairement positifs. Capitalisez sur les aspects plébiscités (souvent liés à la gamification et à la facilité d'utilisation) et continuez d'innover en douceur.")
    elif not avis_negatifs_textes and not avis_positifs_textes:
        recommendations.append("ℹ️ **Aucun avis positif ou négatif distinct** n'a été fourni ou détecté pour générer des recommandations spécifiques. L'analyse se base sur les avis neutres ou l'ensemble des avis si disponibles.")
    return sorted(list(set(recommendations)))

# ### MODIFICATION SIGNIFICATIVE ### Amélioration de la fonction de génération PDF
def generate_pdf(metrics, topics, lxd_results_pdf, theory_recs_pdf):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, page_height = letter
    styles = getSampleStyleSheet()

    # Define margins
    left_margin = 0.75 * inch
    right_margin = 0.75 * inch
    top_margin = 0.75 * inch
    bottom_margin = 0.75 * inch
    content_width = width - left_margin - right_margin

    # Define styles
    title_style = styles['h1']
    title_style.alignment = 1 # Center
    title_style.fontSize = 18
    title_style.leading = 22

    heading_style = styles['h2']
    heading_style.fontSize = 14
    heading_style.leading = 18
    heading_style.spaceBefore = 12
    heading_style.spaceAfter = 6

    body_style = styles['Normal']
    body_style.fontSize = 10
    body_style.leading = 14 # Increased leading for better readability
    body_style.spaceAfter = 6

    list_item_style = styles['Normal']
    list_item_style.fontSize = 9
    list_item_style.leading = 12 # Increased leading
    list_item_style.leftIndent = 0.25 * inch # Indent list items

    story = []
    current_height = page_height - top_margin

    # Helper function to add paragraphs and manage height
    def add_paragraph(text, style, story_list, current_h, page_h, top_m, bottom_m):
        p = Paragraph(text.encode('latin-1', 'replace').decode('latin-1'), style)
        p_w, p_h = p.wrapOn(c, content_width, page_h) # Calculate height
        
        if current_h - p_h < bottom_m: # Check if it fits
            story_list.append(PageBreak())
            current_h = page_h - top_m
        story_list.append(p)
        current_h -= p_h
        return current_h

    # Title
    current_height = add_paragraph("Rapport d'Analyse - DuoScan Pédagogique", title_style, story, current_height, page_height, top_margin, bottom_margin)
    current_height -= 0.2 * inch # Extra space after title

    # Metrics Section
    current_height = add_paragraph("Indicateurs Clés des Sentiments", heading_style, story, current_height, page_height, top_margin, bottom_margin)
    metrics_text = [
        f"Total Avis Analysés: {metrics['Total Avis Analysés']}",
        f"Avis Positifs: {metrics['Avis Positifs 👍']} ({metrics.get('Pos %', 'N/A')})",
        f"Avis Négatifs: {metrics['Avis Négatifs 👎']} ({metrics.get('Neg %', 'N/A')})",
        f"Avis Neutres: {metrics['Avis Neutres 😐']} ({metrics.get('Neu %', 'N/A')})"
    ]
    for mt_text in metrics_text:
        current_height = add_paragraph(mt_text, body_style, story, current_height, page_height, top_margin, bottom_margin)
    current_height -= 0.1 * inch

    # Topics Section
    current_height = add_paragraph("Thèmes Principaux Identifiés (LDA)", heading_style, story, current_height, page_height, top_margin, bottom_margin)
    if topics:
        for topic_text in topics:
            current_height = add_paragraph(f"• {topic_text}", list_item_style, story, current_height, page_height, top_margin, bottom_margin)
    else:
        current_height = add_paragraph("Aucun thème principal détecté.", list_item_style, story, current_height, page_height, top_margin, bottom_margin)
    current_height -= 0.1 * inch

    # LXD Aspects Section
    current_height = add_paragraph("Analyse des Aspects LXD", heading_style, story, current_height, page_height, top_margin, bottom_margin)
    if lxd_results_pdf:
        for aspect, counts in lxd_results_pdf.items():
            aspect_summary = f"• {aspect}: Pos={counts['Positif']}, Nég={counts['Négatif']}, Neu={counts['Neutre']} (Total mentions: {counts['Mentions']})"
            current_height = add_paragraph(aspect_summary, list_item_style, story, current_height, page_height, top_margin, bottom_margin)
    else:
        current_height = add_paragraph("Aucune donnée d'aspect LXD à afficher.", list_item_style, story, current_height, page_height, top_margin, bottom_margin)
    current_height -= 0.1 * inch

    # Recommendations Section
    current_height = add_paragraph("Recommandations Pédagogiques & LXD", heading_style, story, current_height, page_height, top_margin, bottom_margin)
    if theory_recs_pdf:
        for rec_idx, rec_text in enumerate(theory_recs_pdf):
            rec_cleaned = rec_text.replace("🔬 ", "").replace("**", "")
            current_height = add_paragraph(f"{rec_idx + 1}. {rec_cleaned}", list_item_style, story, current_height, page_height, top_margin, bottom_margin)
    else:
        current_height = add_paragraph("Aucune recommandation spécifique générée.", list_item_style, story, current_height, page_height, top_margin, bottom_margin)

    # Build PDF using SimpleDocTemplate for better flow control
    from reportlab.platypus import SimpleDocTemplate, PageBreak
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=left_margin, rightMargin=right_margin,
                            topMargin=top_margin, bottomMargin=bottom_margin)
    try:
        doc.build(story)
    except Exception as e:
        logger.error(f"Erreur majeure lors de la construction du PDF avec SimpleDocTemplate : {e}")
        # Fallback to basic error PDF if build fails
        buffer.seek(0)
        buffer.truncate() # Clear buffer before writing error
        c_error = canvas.Canvas(buffer, pagesize=letter)
        c_error.setFont("Helvetica", 12)
        c_error.drawCentredString(width/2, page_height/2 + 20, "Erreur critique lors de la génération du PDF.")
        error_message_pdf = f"Détail: {str(e)[:100]}"
        c_error.drawCentredString(width/2, page_height/2, error_message_pdf.encode('latin-1', 'replace').decode('latin-1'))
        c_error.save()

    buffer.seek(0)
    return buffer


# --- Sidebar ---
with st.sidebar:
    # ### MODIFICATION ### Photo réintégrée
    st.image("https://i.imgur.com/LF3KIQa.jpeg", width=100, caption="EL FILALI MOHAMED")
    # ### MODIFICATION ### "Version Avancée" enlevé du titre de la sidebar
    st.markdown("# DuoScan Pédagogique 🦉")
    st.divider()
    st.markdown("#### Master : Ingénierie Technopédagogique et Innovation")
    st.markdown("#### Module : Design de l'Expérience d'Apprentissage (LXD)")
    st.markdown("#### Professeur : M. Adil AMMAR")
    st.markdown("#### Réalisé par : EL FILALI Mohamed")
    st.divider()
    st.markdown("### Objectif de l'outil:")
    st.success(
        "Analyser finement le sentiment et les thématiques des retours utilisateurs "
        "de Duolingo (ou similaire) pour identifier avec précision des pistes d'amélioration "
        "basées sur le LXD et les théories de l'apprentissage."
    )
    st.markdown("### Guide Rapide")
    st.markdown("""
    1.  Choisissez la méthode d'entrée (texte ou fichier).
    2.  Fournissez les avis (max 1000 pour performance).
    3.  Cliquez sur "**🚀 Analyser les Avis**".
    4.  Explorez les insights et recommandations générés !
    """)
    st.divider()
    # ### MODIFICATION ### Message de date de mise à jour enlevé
    # st.info(f"Dernière mise à jour du script : {pd.Timestamp('today').strftime('%d/%m/%Y')}")
    st.caption(f"© {pd.Timestamp.now().year} - EL FILALI Mohamed - ITPI")


# --- Main Interface ---
st.title("🦉 DuoScan Pédagogique") # MODIFICATION: Titre simplifié
st.subheader("Analyse des Avis Utilisateurs pour l'Optimisation LXD") # MODIFICATION: Sous-titre simplifié
st.markdown("""
Cet outil permet une analyse sémantique et pédagogique des avis utilisateurs.
Il se base sur :
- Une détection de sentiments sensible aux nuances.
- L'identification de thèmes clés (via LDA) et d'aspects LXD prédominants.
- La génération de recommandations actionnables ancrées dans les théories de l'apprentissage.
""") # MODIFICATION: Description simplifiée
st.divider()

# --- Data Input Section ---
st.header("📥 1. Soumission des Avis Utilisateurs")
input_method = st.radio(
    "Comment souhaitez-vous fournir les avis ?",
    ("Coller le texte directement", "Télécharger un fichier (.txt, .csv, .xls, .xlsx)"),
    key="input_method_choice",
    horizontal=True
)

avis_entres_bruts = []
MAX_REVIEWS_HARD_LIMIT = 2000
MAX_REVIEWS_DEFAULT_ANALYSIS = 1000

if input_method == "Coller le texte directement":
    avis_texte_area = st.text_area(
        f"Collez ici les avis (un par ligne, {MAX_REVIEWS_DEFAULT_ANALYSIS} recommandés, {MAX_REVIEWS_HARD_LIMIT} max.) :",
        height=200, key="text_area_input",
        placeholder=("Exemple 1 : Duolingo m'a vraiment aidé à apprendre l'espagnol, c'est ludique et efficace !\n"
                     "Exemple 2 : Je suis déçu, l'application plante souvent et les publicités sont trop présentes, c'est frustrant.\n"
                     "Exemple 3 : C'est pas mal mais le système de vies est un peu décourageant parfois.")
    )
    if avis_texte_area:
        avis_entres_bruts = [avis.strip() for avis in avis_texte_area.split('\n') if avis.strip()]

elif input_method.startswith("Télécharger un fichier"):
    uploaded_file = st.file_uploader(
        f"Sélectionnez un fichier .txt, .csv, .xls ou .xlsx (colonne 'avis', 'review', 'commentaire' ou unique colonne). Limite : {MAX_REVIEWS_HARD_LIMIT} avis.",
        type=["txt", "csv", "xls", "xlsx"], key="file_uploader_input"
    )
    if uploaded_file is not None:
        try:
            filename = uploaded_file.name.lower()
            df_avis = None # Initialize df_avis
            if filename.endswith('.txt'):
                encoding = detect_encoding(uploaded_file)
                avis_entres_bruts = [line.decode(encoding).strip() for line in uploaded_file if line.decode(encoding).strip()]
            elif filename.endswith('.csv'):
                encoding = detect_encoding(uploaded_file)
                try: df_avis = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
                except Exception as e_csv:
                    st.error(f"Erreur de lecture CSV : {e_csv}. Vérifiez l'encodage ({encoding} détecté) et le séparateur.")
                    st.stop()
            elif filename.endswith(('.xls', '.xlsx')):
                try: df_avis = pd.read_excel(uploaded_file, engine='openpyxl' if filename.endswith('.xlsx') else 'xlrd')
                except Exception as e_excel:
                    st.error(f"Erreur de lecture Excel : {e_excel}. Assurez-vous que le fichier n'est pas corrompu.")
                    st.stop()
            
            if df_avis is not None: # Process if df_avis was loaded
                potential_cols = ['avis', 'review', 'reviews', 'commentaire', 'commentaires', 'text', 'content']
                review_col = None
                for col_name in df_avis.columns: # Iterate through actual column names
                    if str(col_name).lower() in potential_cols:
                        review_col = col_name
                        break
                
                if review_col:
                    avis_entres_bruts = [str(avis).strip() for avis in df_avis[review_col].dropna().tolist() if str(avis).strip()]
                elif len(df_avis.columns) == 1:
                    avis_entres_bruts = [str(avis).strip() for avis in df_avis.iloc[:, 0].dropna().tolist() if str(avis).strip()]
                else:
                    st.error(f"Fichier {filename.split('.')[-1].upper()} : impossible de trouver une colonne d'avis pertinente. Veuillez nommer la colonne d'avis de manière explicite ou utiliser un fichier à une seule colonne.")
                    st.stop()
        except Exception as e:
            st.error(f"⚠️ Erreur majeure lors de la lecture du fichier '{uploaded_file.name}' : {e}. Vérifiez le format et l'encodage (UTF-8 recommandé).")
            st.stop()

st.divider()
num_reviews_to_analyze = MAX_REVIEWS_DEFAULT_ANALYSIS
if avis_entres_bruts:
    max_slider_val = min(len(avis_entres_bruts), MAX_REVIEWS_HARD_LIMIT)
    if max_slider_val > 1:
        num_reviews_to_analyze = st.slider(
            f"Nombre d'avis à analyser (sur {len(avis_entres_bruts)} détectés) :", 1, max_slider_val,
            min(MAX_REVIEWS_DEFAULT_ANALYSIS, max_slider_val),
            step=10 if max_slider_val > 100 else 1,
            help=f"Ajustez pour équilibrer la profondeur de l'analyse et le temps de traitement. Limite stricte à {MAX_REVIEWS_HARD_LIMIT}."
        )
    else: num_reviews_to_analyze = max_slider_val

if st.button(f"🚀 Analyser {num_reviews_to_analyze if avis_entres_bruts else ''} Avis", type="primary", use_container_width=True, key="analyze_button"):
    if not avis_entres_bruts:
        st.warning("⚠️ Veuillez fournir des avis avant de lancer l'analyse.")
        st.stop()
    avis_a_analyser = avis_entres_bruts[:num_reviews_to_analyze]
    if len(avis_entres_bruts) > num_reviews_to_analyze:
        st.info(f"Analyse limitée aux {num_reviews_to_analyze} premiers avis sur {len(avis_entres_bruts)} fournis.")

    st.header("📊 2. Résultats de l'Analyse")
    progress_bar = st.progress(0, text="Initialisation de l'analyse...")
    with st.spinner("🦉 Analyse sémantique et thématique en cours..."):
        avis_analyses = []
        sentiments_counts = {"Positif 👍": 0, "Négatif 👎": 0, "Neutre 😐": 0}
        avis_positifs_textes, avis_negatifs_textes, avis_neutres_textes = [], [], []
        total_avis_pour_analyse = len(avis_a_analyser)
        for i, avis_texte in enumerate(avis_a_analyser):
            progress_text = f"Traitement de l'avis {i+1}/{total_avis_pour_analyse}..."
            progress_bar.progress((i + 1) / total_avis_pour_analyse, text=progress_text)
            sentiment = analyser_sentiment(avis_texte)
            avis_analyses.append({"id": i + 1, "texte_original": avis_texte, "sentiment_detecte": sentiment})
            sentiments_counts[sentiment] += 1
            if sentiment == "Positif 👍": avis_positifs_textes.append(avis_texte)
            elif sentiment == "Négatif 👎": avis_negatifs_textes.append(avis_texte)
            else: avis_neutres_textes.append(avis_texte)
        progress_bar.progress(1.0, text="Analyse des sentiments terminée ! 🎉")
        st.toast('Analyse des sentiments terminée ! 🎉', icon='✅')

        st.subheader("📈 Vue d'Ensemble des Sentiments")
        total_analyzed = len(avis_analyses)
        metrics_for_pdf = {
            "Total Avis Analysés": total_analyzed,
            "Avis Positifs 👍": sentiments_counts["Positif 👍"],
            "Avis Négatifs 👎": sentiments_counts["Négatif 👎"],
            "Avis Neutres 😐": sentiments_counts["Neutre 😐"],
            "Pos %": f"{((sentiments_counts['Positif 👍']/total_analyzed)*100):.1f}%" if total_analyzed > 0 else "0%",
            "Neg %": f"{((sentiments_counts['Négatif 👎']/total_analyzed)*100):.1f}%" if total_analyzed > 0 else "0%",
            "Neu %": f"{((sentiments_counts['Neutre 😐']/total_analyzed)*100):.1f}%" if total_analyzed > 0 else "0%",
        }
        cols_metriques = st.columns(4)
        cols_metriques[0].metric("Avis Analysés", metrics_for_pdf["Total Avis Analysés"])
        cols_metriques[1].metric("Avis Positifs 👍", metrics_for_pdf["Avis Positifs 👍"], delta=metrics_for_pdf["Pos %"])
        cols_metriques[2].metric("Avis Négatifs 👎", metrics_for_pdf["Avis Négatifs 👎"], delta=metrics_for_pdf["Neg %"])
        cols_metriques[3].metric("Avis Neutres 😐", metrics_for_pdf["Avis Neutres 😐"], delta=metrics_for_pdf["Neu %"])

        if total_analyzed > 0:
            df_sentiments = pd.DataFrame(list(sentiments_counts.items()), columns=['Sentiment', 'Nombre'])
            sentiment_color_map = {"Positif 👍": "#28a745", "Négatif 👎": "#dc3545", "Neutre 😐": "#6c757d"}
            fig = px.pie(df_sentiments, names='Sentiment', values='Nombre', title="Répartition des Sentiments",
                         color='Sentiment', color_discrete_map=sentiment_color_map, hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label+value')
            fig.update_layout(showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend_title_text='Catégories')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Aucune donnée de sentiment à afficher.")
        st.divider()
        
        all_avis_textes_analyzed = [avis['texte_original'] for avis in avis_analyses]
        progress_bar.progress(0.33, text="Détection des thèmes principaux (LDA)...")
        with st.spinner("Identification des thèmes majeurs..."):
            detected_topics_list = detect_topics(all_avis_textes_analyzed, num_topics=5, n_top_words=7)
        progress_bar.progress(0.66, text="Analyse des aspects LXD...")
        with st.spinner("Analyse des aspects LXD..."):
            sentiments_list_for_lxd = [entry["sentiment_detecte"] for entry in avis_analyses]
            lxd_results = analyze_lxd_aspects(all_avis_textes_analyzed, sentiments_list_for_lxd)
        progress_bar.progress(1.0, text="Analyses thématiques et LXD terminées !")

        with st.expander("🔍 Exploration Détaillée des Avis Filtrés", expanded=False):
            col_filter1, col_filter2 = st.columns(2)
            sentiment_filter = col_filter1.selectbox("Filtrer par Sentiment", ["Tous", "Positif 👍", "Négatif 👎", "Neutre 😐"], key="sentiment_filter_exp")
            topic_options_display = ["Tous"] + [f"{t.split(':')[0]} ({t.split(': ')[1][:30]}...)" for t in detected_topics_list]
            topic_filter_display = col_filter2.selectbox("Filtrer par Mots-Clés d'un Thème", topic_options_display, key="topic_filter_exp")
            filtered_avis_df = pd.DataFrame(avis_analyses)
            if sentiment_filter != "Tous":
                filtered_avis_df = filtered_avis_df[filtered_avis_df["sentiment_detecte"] == sentiment_filter]
            if topic_filter_display != "Tous":
                actual_topic_prefix = topic_filter_display.split(" (")[0]
                selected_topic_full = next((t for t in detected_topics_list if t.startswith(actual_topic_prefix)), None)
                if selected_topic_full:
                    topic_keywords = [kw.strip() for kw in selected_topic_full.split(": ")[1].split(",")]
                    filtered_avis_df = filtered_avis_df[
                        filtered_avis_df["texte_original"].str.lower().apply(lambda x: any(kw in x for kw in topic_keywords))
                    ]
            if not filtered_avis_df.empty:
                st.dataframe(
                    filtered_avis_df[['id', 'sentiment_detecte', 'texte_original']].rename(
                        columns={'id':'ID', 'sentiment_detecte':'Sentiment', 'texte_original':'Avis Original'}),
                    use_container_width=True, hide_index=True, height=350
                )
            else: st.info("Aucun avis ne correspond à vos critères de filtrage.")
        st.divider()

        st.subheader("☁️ Nuages de Mots : Termes Fréquents par Sentiment")
        if not avis_positifs_textes and not avis_negatifs_textes and not avis_neutres_textes:
            st.info("Pas assez de données textuelles pour générer les nuages de mots.")
        else:
            tab_positif, tab_negatif, tab_neutre = st.tabs(["🟢 Termes Positifs", "🔴 Termes Négatifs", "⚪ Termes Neutres"])
            with tab_positif:
                if avis_positifs_textes: generer_nuage_mots(avis_positifs_textes, "issus des Avis Positifs")
                else: st.info("Aucun avis positif pour générer un nuage de mots.")
            with tab_negatif:
                if avis_negatifs_textes: generer_nuage_mots(avis_negatifs_textes, "issus des Avis Négatifs")
                else: st.info("Aucun avis négatif pour générer un nuage de mots.")
            with tab_neutre:
                if avis_neutres_textes: generer_nuage_mots(avis_neutres_textes, "issus des Avis Neutres")
                else: st.info("Aucun avis neutre pour générer un nuage de mots.")
        st.divider()
        
        st.subheader("🎯 Thèmes Principaux Identifiés dans les Avis (via LDA)")
        if detected_topics_list:
            st.markdown("Les thèmes suivants ont été extraits automatiquement des avis analysés :")
            for topic_item in detected_topics_list: st.markdown(f"- {topic_item}")
        else: st.warning("Analyse thématique non concluante : pas assez de données.")
        st.divider()

        st.subheader("🖌️ Analyse Détaillée des Aspects LXD")
        if lxd_results:
            st.markdown("Répartition des sentiments pour les aspects LXD identifiés (nombre de mentions) :")
            lxd_df_list = []
            for aspect, counts in lxd_results.items():
                if counts["Mentions"] > 0:
                     lxd_df_list.append({'Aspect LXD': aspect, 'Positif': counts['Positif'], 'Négatif': counts['Négatif'], 'Neutre': counts['Neutre'], 'Total Mentions': counts['Mentions']})
            if lxd_df_list:
                df_lxd_display = pd.DataFrame(lxd_df_list).sort_values(by="Total Mentions", ascending=False)
                st.dataframe(df_lxd_display, hide_index=True, use_container_width=True)
                if not df_lxd_display.empty:
                    df_lxd_melted = df_lxd_display.melt(id_vars=['Aspect LXD', 'Total Mentions'], value_vars=['Positif', 'Négatif', 'Neutre'], var_name='Sentiment', value_name='Nombre d\'Avis')
                    df_lxd_melted = df_lxd_melted[df_lxd_melted['Nombre d\'Avis'] > 0]
                    if not df_lxd_melted.empty:
                        fig_lxd = px.bar(df_lxd_melted, x='Aspect LXD', y='Nombre d\'Avis', color='Sentiment', barmode='group', title="Ventilation des Sentiments par Aspect LXD", color_discrete_map=sentiment_color_map, labels={'Nombre d\'Avis': "Nombre d'Avis Mentionnant l'Aspect"})
                        fig_lxd.update_layout(xaxis_tickangle=-45, yaxis_title="Nombre d'Avis", xaxis_title="Aspect LXD")
                        st.plotly_chart(fig_lxd, use_container_width=True)
                    else: st.info("Aucune mention avec sentiment spécifique pour les aspects LXD à visualiser.")
            else: st.info("Aucun aspect LXD spécifique n'a pu être clairement identifié.")
        else: st.warning("L'analyse LXD n'a pas pu être effectuée.")
        st.divider()

        st.header("💡 3. Synthèse Interprétative et Recommandations Stratégiques")
        st.markdown("#### Analyse Quantitative et Qualitative Automatisée")
        positive_percentage = (sentiments_counts["Positif 👍"] / total_analyzed * 100) if total_analyzed else 0
        negative_percentage = (sentiments_counts["Négatif 👎"] / total_analyzed * 100) if total_analyzed else 0
        if positive_percentage > 65: st.success(f"✅ **Perception Globale Très Positive ({positive_percentage:.1f}%)**")
        elif positive_percentage > 40: st.info(f"👍 **Perception Globale Positive ({positive_percentage:.1f}%)**")
        else: st.warning(f"⚠️ **Perception Mitigée ou Négative ({positive_percentage:.1f}% de positifs)**")
        if negative_percentage > 30: st.error(f"🚨 **Frustrations Significatives ({negative_percentage:.1f}%)**")
        elif negative_percentage > 15: st.warning(f"🔍 **Points de Friction Présents ({negative_percentage:.1f}%)**")
        else: st.success(f"😌 **Faible Taux de Négativité ({negative_percentage:.1f}%)**")

        st.markdown("#### Recommandations Pédagogiques et LXD (Basées sur les Théories de l'Apprentissage)")
        if avis_negatifs_textes or avis_positifs_textes or lxd_results:
            with st.spinner("Élaboration des recommandations stratégiques..."):
                theory_recommendations = generate_theory_recommendations(avis_negatifs_textes, avis_positifs_textes, lxd_results)
            if theory_recommendations:
                st.info("Les recommandations suivantes sont générées automatiquement...")
                for i, rec in enumerate(theory_recommendations): st.markdown(f"**{i+1}.** {rec}")
            else: st.info("Aucune recommandation spécifique n'a pu être générée automatiquement.")
        else: st.info("Pas assez d'avis pour générer des recommandations pédagogiques robustes.")
        st.divider()

        st.markdown("### 📥 Exportation des Données et du Rapport")
        col1_exp, col2_exp = st.columns(2)
        with col1_exp:
            if avis_analyses:
                df_export = pd.DataFrame(avis_analyses)[['id', 'sentiment_detecte', 'texte_original']].rename(columns={'id':'ID', 'sentiment_detecte':'Sentiment Détecté', 'texte_original':'Texte de l\'Avis'})
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Télécharger les Avis Analysés (CSV)", data=csv_data, file_name=f"resultats_analyse_avis_duolingo_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        with col2_exp:
            if avis_analyses:
                lxd_results_for_pdf = {k: v for k, v in lxd_results.items() if v["Mentions"] > 0} if 'lxd_results' in locals() and lxd_results else {}
                theory_recs_for_pdf = theory_recommendations if 'theory_recommendations' in locals() and theory_recommendations else []
                pdf_buffer = generate_pdf(metrics_for_pdf, detected_topics_list, lxd_results_for_pdf, theory_recs_for_pdf)
                st.download_button(label="📄 Télécharger le Rapport de Synthèse (PDF)", data=pdf_buffer, file_name=f"rapport_synthese_duoscan_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
        st.divider()

        st.markdown("### ✍️ Votre Espace d'Analyse Stratégique")
        st.markdown("En vous appuyant sur les données générées, veuillez élaborer votre propre analyse...")
        st.text_area(
            "Rédigez ici votre analyse approfondie et vos recommandations personnalisées pour M. AMMAR :",
            height=300, key="interpretation_area_detailed_final",
            placeholder=("Exemple : L'analyse révèle une forte corrélation entre les avis négatifs sur 'le système de vies'...")
        )
else:
    st.info("ℹ️ Prêt pour une analyse approfondie des avis ? Soumettez vos données et cliquez sur 'Analyser'.")
    st.markdown("---")
    st.markdown("### À propos de DuoScan Pédagogique") # MODIFICATION: "Avancé" enlevé
    st.markdown("""
    Cette version de **DuoScan Pédagogique** vise à fournir une analyse fine et actionnable des retours utilisateurs.
    Elle met l'accent sur :
    - Une détection de sentiment robuste.
    - Une identification claire des aspects LXD et de leur perception.
    - Des recommandations ancrées dans les théories de l'apprentissage et les données concrètes des avis.
    L'objectif est de transformer les avis bruts en intelligence stratégique pour l'amélioration continue des expériences d'apprentissage.
    """) # MODIFICATION: Description simplifiée