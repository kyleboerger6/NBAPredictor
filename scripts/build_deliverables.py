"""Create an EDITABLE presentation and a DRAFT report from real run_all.py results.
Use only after a complete, verified run; report should be reviewed/rewritten by the group.
python scripts/build_deliverables.py
"""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
RES=ROOT/'results'
DEST=ROOT/'deliverables'
DEST.mkdir(exist_ok=True)
summary=json.loads((RES/'summary.json').read_text(encoding='utf8'))
if summary['status']!='completed':
    raise RuntimeError('Missing SHAP/LIME: please run `python run_all.py` without --skip-explainers first.')
metrics=pd.read_csv(RES/'04_classification_metrics.csv')
cls=metrics[metrics['split']=='test'].set_index('model')
upsets=pd.read_csv(RES/'05_upset_metrics.csv').set_index('model')
prevalence=pd.read_csv(RES/'05_upset_prevalence.csv').set_index('split')
imp=pd.read_csv(RES/'06_permutation_importance.csv')
shap=pd.read_csv(RES/'06_shap_importance.csv')
clust=pd.read_csv(RES/'07_clustering_comparison.csv').set_index('model')
season=pd.read_csv(RES/'08_season_metrics.csv').set_index('season')
cal=pd.read_csv(RES/'08_calibration.csv')
selected=summary['selected_model']

# ----- PPTX -----------------------------------------------------------------
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

NAVY=RGBColor(15,24,40); SLATE=RGBColor(29,43,63); WHITE=RGBColor(247,249,252)
TEAL=RGBColor(59,222,185); MUTED=RGBColor(170,188,210); ORANGE=RGBColor(255,184,100)
prs=Presentation(); prs.slide_width=Inches(13.33); prs.slide_height=Inches(7.5)

def shape(slide, x,y,w,h, fill, radius=False):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                             Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid();sh.fill.fore_color.rgb=fill;sh.line.fill.background();return sh

def txt(slide,text,x,y,w,h,pt=22,color=WHITE,bold=False,align=None):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame
    tf.word_wrap=True;tf.margin_left=Inches(.02);tf.margin_top=Inches(.02)
    for i,line in enumerate(str(text).split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
        if align is not None: p.alignment=align
        p.space_after=Pt(8)
        for r in p.runs:r.font.name='Aptos';r.font.size=Pt(pt);r.font.bold=bold;r.font.color.rgb=color
    return box

def base(title,sub=''):
    slide=prs.slides.add_slide(prs.slide_layouts[6]);shape(slide,0,0,13.33,7.5,NAVY)
    shape(slide,.43,.48,.11,.66,TEAL)
    txt(slide,title,.71,.48,11.9,.65,32,WHITE,True)
    if sub:txt(slide,sub,.73,1.22,11.9,.52,15,MUTED)
    shape(slide,.7,7.10,11.92,.012,SLATE)
    txt(slide,f'FSU CIS 4930  |  NBA Match Predictor  |  {len(prs.slides):02d}',.72,7.14,10.5,.2,10,MUTED)
    return slide

def text_panel(slide,headline,items,x=.74,y=2,w=11.8,font_size=19,step=.82):
    shape(slide,x,y,w,max(2.5,1.1+len(items)*step),SLATE,True)
    txt(slide,headline,x+.35,y+.25,w-.7,.55,23,TEAL,True)
    yy=y+.98
    for item in items:
        txt(slide,'• '+item,x+.37,yy,w-.8,step-.08,font_size,WHITE)
        yy+=step

def metric_card(slide,label,value,x,y,w=2.8):
    shape(slide,x,y,w,1.52,SLATE,True)
    txt(slide,label,x+.17,y+.22,w-.34,.39,15,MUTED)
    txt(slide,value,x+.15,y+.68,w-.3,.6,27,TEAL,True)

def img(slide, filename, x,y,w,h):
    from PIL import Image
    p=RES/'figures'/filename
    if not p.exists():return
    with Image.open(p) as im:iw,ih=im.size
    ratio=min(w/iw,h/ih); pw=iw*ratio;ph=ih*ratio
    slide.shapes.add_picture(str(p), Inches(x+(w-pw)/2),Inches(y+(h-ph)/2),width=Inches(pw),height=Inches(ph))

s=base('Predicting NBA Games Before Tipoff','Mitigating imbalance in NBA predictions via interpretable ensemble models')
txt(s,'Kyle Boerger  |  Shezan Chisty  |  Alejandro Salazar',.8,2.2,11.9,1,24,WHITE,True)
shape(s,.78,3.45,11.8,1.5,SLATE,True)
txt(s,'A reproducible study of winners, underdog upsets, feature importance, and team archetypes',1.18,3.78,11.0,.82,24,TEAL)

s=base('Research questions','Beyond simply reporting classification accuracy')
text_panel(s,'What can historical information explain?',["Can pre-game statistics predict winners more accurately than a home-team baseline?",
   "Can imbalance remedies improve identification of real underdog upsets?",
   "Which pre-game factors influence predictions, and do clustered team archetypes help?"],y=1.82)

s=base('Data, features, and leakage safeguards','Kaggle: Nathan Lauga, NBA games data | regular-season games')
metric_card(s,'Clean games',f"{summary['dataset_games_after_cleaning']:,}",.78,1.95)
metric_card(s,'Usable games',f"{summary['games_after_feature_generation']:,}",3.83,1.95)
metric_card(s,'Predictors','54',6.88,1.95)
metric_card(s,'Hold-out test','3,225',9.93,1.95)
text_panel(s,'Pre-game by construction',["10-game rolling means and season statistics shift by one game",
  "Elo values recorded before each game's result updates ratings",
  "Train: 2003-16 | Validation: 2017-18 | Test: 2019-21; no random shuffle"],y=3.60)

s=base('The single pipeline','One dataset, many experiments')
for i,(a,b) in enumerate([('01','Clean NBA games'),('02','Rolling + Elo pregame features'),('03','Chronological split'),('04','Model training'),('05','Additional investigations')]):
    x=.73+i*2.55
    shape(s,x,2.16,2.39,2.2,SLATE,True);txt(s,a,x+.19,2.4,1.95,.6,28,TEAL,True);txt(s,b,x+.2,3.06,2.05,.93,18)
txt(s,'Pre-game data only  •  Training-only preprocessing  •  Validation model selection  •  Held-out reporting',.76,5.46,11.9,1.1,22,WHITE)

s=base('NBA winner prediction','Test accuracy: held-out 2019–2021 NBA seasons')
for i,(m,label) in enumerate([('always_home','Home baseline'),('logistic','Logistic'),('random_forest','Random Forest'),('xgboost','XGBoost')]):
    metric_card(s,label,f"{100*cls.loc[m,'accuracy']:.2f}%",.74+i*3.10,2.02,2.94)
txt(s,f'Chosen model: {selected} (selected using VALIDATION ROC AUC, not test accuracy).',.82,4.35,11.6,1,23,TEAL,True)
txt(s,'XGBoost happened to have slightly higher test accuracy; the declared selection rule still chooses Logistic Regression.',.82,5.41,11.5,.96,18,MUTED)

s=base('Classification results','Selected Logistic Regression; actual prediction counts')
img(s,'04_confusion.png',.8,1.72,7.0,4.91)
metric_card(s,'Test ROC AUC',f"{cls.loc['logistic','roc_auc']:.3f}",9.04,2.11,3.35)
metric_card(s,'Test Brier score',f"{cls.loc['logistic','brier']:.3f}",9.04,4.00,3.35)

s=base('The harder task: detecting upsets','An upset = clear Elo underdog wins; not simply an away win')
metric_card(s,'Upset prevalence',f"{100*prevalence.loc['test','upset_rate']:.1f}%",.77,1.92,3.78)
metric_card(s,'Always-upset F1',f"{upsets.loc['always_predict_upset','f1']:.3f}",4.79,1.92,3.78)
metric_card(s,'Weighted-model F1',f"{upsets.loc['class_weight_balanced','f1']:.3f}",8.81,1.92,3.78)
text_panel(s,'The real finding',["Never predicting an upset gives zero upset recall (F1 = 0)",
 "Always predicting an upset has 100% recall but low precision",
 f"Class weighting barely beats always-upset F1 (+{upsets.loc['class_weight_balanced','f1']-upsets.loc['always_predict_upset','f1']:.3f})"],y=3.55)

s=base('Why did the model predict a winner?','Descriptive importance; not proof of causation')
img(s,'06_permutation.png',.73,1.60,8.4,5.12)
text_panel(s,'Cross-checks',["Forest impurity ranking", "Permutation on held-out games",
 "SHAP global contributions", "LIME: 3 local games"],x=9.19,y=1.91,w=3.4,font_size=15,step=.73)

s=base('Do team archetypes help?','K-Means on training-era pre-game team statistics; K=3')
metric_card(s,'RF without clusters',f"{100*clust.loc['RF_without_archetypes','accuracy']:.2f}%",.82,2.0,5.6)
metric_card(s,'RF with clusters',f"{100*clust.loc['RF_with_archetypes','accuracy']:.2f}%",6.90,2.0,5.6)
shape(s,.79,4.05,11.8,1.86,SLATE,True)
txt(s,'Only one additional correct prediction out of 3,225 test games.',1.14,4.49,10.96,.77,22,TEAL,True)
txt(s,'Interpret clustering as exploratory, not a demonstrated improvement.',1.14,5.45,10.99,.40,17,MUTED)

s=base('Reliability across seasons','No recalibration on the test set; purely diagnostic')
img(s,'08_calibration.png',.75,1.72,6.6,4.82)
shape(s,7.66,1.82,4.89,4.78,SLATE,True)
txt(s,'Selected model accuracy',8.0,2.05,4.18,.58,23,TEAL,True)
for j,(year,r) in enumerate(season.iterrows()):
    txt(s,f'{int(year)}–{(int(year)+1)%100:02d}:   {100*r.accuracy:.1f}%',8.04,2.98+j*.69,4.0,.5,21)
txt(s,f'Overall Brier: {cls.loc[selected,"brier"]:.3f}',8.04,5.56,4.0,.45,19,MUTED)

s=base('Conclusions and limitations','A well-designed experiment can produce useful negative findings')
text_panel(s,'What we learned',["True pre-game predictions outperform a trivial home-team baseline",
 "Upset detection is difficult; different operating thresholds change precision/recall",
 "Elo and recent scoring performance are informative; archetype gain is negligible",
 "Results vary by season; no injuries, roster changes, or live data are modeled"],y=1.76)

s=base('Questions / reproducibility','What we can demonstrate if asked')
text_panel(s,'Run and inspect',["python -m pytest -q",
 "python run_all.py",
 "results/summary.json plus per-experiment CSV tables and figures",
 "Review no-leakage tests, the underdog definition, and validation-only selection"],y=1.83)

slides=DEST/'NBA_Project_Results_Presentation.pptx';prs.save(slides)
print('Presentation ->',slides)

# ----- PDF report (editable source: this generator + code) -------------------
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                KeepTogether, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from xml.sax.saxutils import escape

pdf=DEST/'NBA_Project_Results_Report_DRAFT.pdf'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyT', parent=styles['BodyText'], fontName='Helvetica', fontSize=9.4,
                          leading=13, spaceAfter=7))
styles.add(ParagraphStyle(name='HeadT', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12.3,
                          leading=16, textColor=colors.HexColor('#173149'), spaceBefore=12, spaceAfter=7))
styles.add(ParagraphStyle(name='SmallT', parent=styles['BodyText'], fontName='Helvetica', fontSize=8.5, leading=11))
styles.add(ParagraphStyle(name='TitleT', parent=styles['Title'], fontSize=16, leading=20, textColor=colors.HexColor('#173149')))
story=[]
def para(text,kind='BodyT'): story.append(Paragraph(text,styles[kind]))
def head(text):para(text,'HeadT')
def table(columns,rows,widths=None):
    data=[[Paragraph(escape(str(x)),styles['SmallT']) for x in columns]]
    for row in rows:data.append([Paragraph(escape(str(x)),styles['SmallT']) for x in row])
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DBEEE8')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F9')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),
        ('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.grey)]))
    story.append(t);story.append(Spacer(1,9))
def figure(name, caption, w=6.3):
    p=RES/'figures'/name
    from PIL import Image as PILImage
    with PILImage.open(p) as im:iw,ih=im.size
    h=min(3.3,w*ih/iw)
    story.append(Image(str(p),width=inch*w,height=inch*h))
    para(escape(caption),'SmallT')

para('Mitigating Imbalance in NBA Predictions via Interpretable Ensemble Models','TitleT')
para('Kyle Boerger  |  Shezan Chisty  |  Alejandro Salazar<br/>CIS 4930 - Data Mining, Florida State University','SmallT')
para('<b>Report status:</b> Results-based DRAFT generated from the current code and data. The team must verify all findings, citations, explanations, and academic-integrity requirements before submitting.','SmallT')
head('Abstract')
para(f'This study examines pre-game NBA winner prediction as well as three data mining questions beyond standard classification: detecting underdog upsets under class imbalance, interpreting influential predictive variables, and clustering teams into statistical archetypes. Historical regular-season games are transformed into leakage-safe rolling and Elo-based features. Models are selected using chronological validation years, then evaluated on unseen 2019-2021 seasons. The validation-selected {escape(selected)} model obtains {100*cls.loc[selected,"accuracy"]:.2f}% test accuracy versus {100*cls.loc["always_home","accuracy"]:.2f}% for always picking the home team. Secondary experiments find recall/precision tradeoffs in upset detection, limited incremental predictive value from team archetypes, and some variation across seasons.')
head('1. Problem, motivation, and literature')
para('The objective is to estimate the probability that the home team wins a future matchup using only information known before tipoff. Comparing classification accuracy alone would not resolve the challenge identified in the instructor\'s feedback. Our additional questions are whether clear underdog wins form a minority target that can be predicted, which pre-game statistical factors drive model decisions, and whether unsupervised archetype information is useful.')
para('NBA result-prediction literature includes early neural-network approaches (Loeffelholz, Bednar &amp; Bauer, 2009) and feature-based machine learning (Thabtah, Zhang &amp; Abdelhamid, 2019; Chen et al., 2021). These studies motivate using diverse team performance measurements rather than relying on only recent wins. However, accuracy claims from different studies are not directly comparable when the forecast point, time periods, input availability, or evaluation splits differ. Our study standardizes the forecasting point before tipoff and reserves future seasons as the final test.')
para('Random Forests aggregate many decision trees (Breiman, 2001), allowing nonlinear interactions among performance variables; Logistic Regression provides a transparent linear baseline. XGBoost supplies a second form of tree ensemble, making it useful to compare algorithm families on identical pregame features. Importantly, the goal is not to claim one approach must always win, but to compare results on the same historical split and then analyze their error patterns.')
para('Class imbalance can make accuracy look strong while minority cases are systematically missed. SMOTE (Chawla et al., 2002) constructs synthetic minority training observations; class weighting instead changes the fitting objective. Our upset experiment distinguishes true pregame underdogs from ordinary away losses and measures recall, precision, and F1 for that explicit minority target. The general team-sport review by Bunker &amp; Susnjak (2022) provides broader motivation for careful evaluation and honest feature engineering.')
para('For interpretability, SHAP (Lundberg &amp; Lee, 2017) and LIME (Ribeiro, Singh &amp; Guestrin, 2016) can describe model predictions at global and local levels. Permutation importance asks whether a feature measurably contributes to held-out predictive performance. These techniques can disagree because correlated features are not independent. Team-archetype clustering is exploratory; a useful representation would improve a held-out predictor, but any gain must be measured rather than assumed. These references and the bibliography should be checked against the original publications before submission.')
head('2. Data and leakage-resistant methodology')
para(f'The input was the Kaggle "NBA games data" historical games.csv archive by Nathan Lauga, filtered to completed regular-season games starting in seasons 2003-2021. Cleaning retained {summary["dataset_games_after_cleaning"]:,} games. After discarding {summary["dataset_games_after_cleaning"]-summary["games_after_feature_generation"]:,} early-season matchups without enough prior games, the modeling set contains {summary["games_after_feature_generation"]:,} rows and 54 numeric predictors.')
para('Each team contributes rolling 10-game box-score means, prior season-to-date performance, rest days, and pre-game Elo. The current game is excluded from each rolling or expanding mean with shift(1), and Elo is recorded before its outcome modifies team ratings. Both current-game box scores and future-season test targets are excluded from the predictive feature matrix. The existing synthetic pipeline tests also perturb a game\'s score to ensure that game\'s own features remain unchanged.')
para('Chronological partition: train on seasons 2003-2016 (16,264 games), validate on 2017-2018 (2,365 games), test on 2019-2021 (3,225 games). Model type is selected using validation ROC AUC; the selected estimator is refit using train plus validation and evaluated on the held-out test. Logistic Regression scales predictors using training-only fitting in its pipeline; tree models require no scaling. All outputs are reproduced from run_all.py.')
story.append(PageBreak())
head('3. Experiment 1 - Pre-game winner classification')
table(['Approach','Test accuracy','ROC AUC','Brier score'],[[x,f'{100*cls.loc[x,"accuracy"]:.2f}%', f'{cls.loc[x,"roc_auc"]:.3f}',f'{cls.loc[x,"brier"]:.3f}'] for x in ['always_home','logistic','random_forest','xgboost']], [2.1*inch,1.1*inch,1.2*inch,1.25*inch])
para(f'The best validation ROC AUC belongs to {escape(selected)}, so the declared selection procedure chooses it, even though another model may have slightly higher test accuracy. Relative to the always-home rule, its accuracy improves by {100*(cls.loc[selected,"accuracy"]-cls.loc["always_home","accuracy"]):.2f} percentage points. The result reflects generalization to later seasons, not perfectly accurate predictions.')
figure('04_confusion.png','Figure 1. Held-out confusion matrix for the validation-selected winner model.',w=4.2)
head('4. Experiment 2 - Underdog upset detection')
para('Upset prediction is a separate target rather than labeling all away wins as upsets. An Elo favorite is defined pre-game as having at least a 65% predicted win probability. We consider only games with a sufficiently strong favorite, and the positive class is an actual underdog win. In the test period, '+f'{int(prevalence.loc["test","upset_count"]):,} of {int(prevalence.loc["test","strong_favorite_games"]):,} qualifying games ({100*prevalence.loc["test","upset_rate"]:.1f}%) were upsets. Classification thresholds were chosen on the validation period, and SMOTE was applied to training data only.')
table(['Method','Test accuracy','Upset precision','Upset recall','Upset F1'],[[x,f'{100*upsets.loc[x,"accuracy"]:.1f}%',f'{upsets.loc[x,"precision"]:.3f}',f'{upsets.loc[x,"recall"]:.3f}',f'{upsets.loc[x,"f1"]:.3f}'] for x in ['never_predict_upset','always_predict_upset','unweighted','class_weight_balanced','SMOTE_train_only']], [1.65*inch,1.0*inch,1.15*inch,1.06*inch,1.0*inch])
para('The never-upset baseline has relatively high accuracy because most favorites win, but fails to identify a single upset (F1=0). Weighting and SMOTE shift the precision-recall tradeoff; their effects cannot be inferred from overall accuracy alone. Predicting many upsets raises recall but causes false alarms, as shown by the low precision. SMOTE does not automatically outperform standard or weighted training.')
false_positives = int(((pd.read_csv(RES/'05_upset_predictions.csv')['actual_upset'] == 0) &
    (pd.read_csv(RES/'05_upset_predictions.csv')['predict_class_weight_balanced'] == 1)).sum())
para(f'An essential additional baseline predicts every qualifying game as an upset. This achieves 100% recall but only {upsets.loc["always_predict_upset","precision"]:.3f} precision and {upsets.loc["always_predict_upset","f1"]:.3f} F1. The class-weighted model reaches F1={upsets.loc["class_weight_balanced","f1"]:.3f}, a gain of just {upsets.loc["class_weight_balanced","f1"]-upsets.loc["always_predict_upset","f1"]:.3f}, while producing {false_positives:,} false upset predictions. The tiny gain shows that high recall alone is not evidence of effective underdog discrimination.')
figure('05_upsets.png','Figure 2. Held-out upset-detection F1 across comparison methods.',w=5.0)
head('5. Experiment 3 - Feature importance and local explanations')
para('We compare three global views: Random Forest impurity importance, held-out permutation importance scored by ROC AUC change, and mean absolute SHAP attributions; LIME explains several individual decisions. They answer slightly different questions and can disagree when inputs are correlated. Importance is association with model predictions, not evidence of causal effects on a real basketball match.')
table(['Permutation feature','ROC AUC reduction'],[[str(x.feature),f'{x.importance:.4f}'] for _,x in imp.head(8).iterrows()], [3.85*inch,1.95*inch])
para('Pre-game Elo probability, Elo difference, and recent performance margin appear among the strongest permutation features in this limited held-out sample. Because permutation importance uses only a subset of test matchups and a few repeats, low-ranked results should not be over-interpreted. SHAP and LIME output tables are provided for follow-up inspection.')
figure('06_permutation.png','Figure 3. Top permutation feature importances in held-out NBA games.',w=5.1)
figure('06_shap.png','Figure 4. SHAP cross-check using mean absolute attributions from Random Forest.',w=5.1)
head('6. Experiment 4 - K-Means archetype experiment')
para('K-Means with K=3 clusters pre-game team-game profiles from historical offensive and defensive statistics. The feature scaler and clustering centroids are fitted on training-era games only, then applied unchanged to validation and test games. Home/away cluster assignments are appended as extra one-hot features to a Random Forest. A separate unaugmented Random Forest with comparable settings provides the baseline.')
a=clust.loc['RF_without_archetypes','accuracy'];b=clust.loc['RF_with_archetypes','accuracy'];diff=int(round((b-a)*3225))
para(f'Test accuracy changes from {100*a:.2f}% to {100*b:.2f}%, corresponding to {diff:+d} additional correct games out of 3,225. This is negligible evidence of incremental predictive value. The clusters may summarize team style, but should not be asserted to represent objectively correct archetypes or an established performance gain.')
figure('07_archetypes.png','Figure 5. Accuracy comparison for Random Forest with and without K-Means archetypes.',w=4.35)
head('7. Experiment 5 - Calibration and seasonal changes')
para(f'Probability reliability is examined without calibrating on held-out test outcomes. The validation-selected {escape(selected)} model has test Brier score {cls.loc[selected,"brier"]:.3f}. A reliability diagram compares forecast probability bins with the observed fraction of home wins. This diagnostic assesses confidence quality and does not prove causation or long-term stability.')
table(['Test season','Games','Accuracy','ROC AUC'],[[f'{int(year)}-{(int(year)+1)%100:02d}',int(r.n),f'{100*r.accuracy:.1f}%',f'{r.roc_auc:.3f}'] for year,r in season.iterrows()],[1.6*inch,1.0*inch,1.4*inch,1.3*inch])
para('Performance varies across seasons, including the unusual 2019-2020 disrupted season. Such variation may reflect roster changes, league-wide shifts, and missing contextual predictors, but this experiment alone does not identify a causal explanation.')
figure('08_calibration.png','Figure 6. Reliability of the validation-selected model on held-out NBA games.',w=4.2)
head('8. Implementation details, limitations, and conclusion')
para('Implementation is in Python with pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, SHAP, LIME, and matplotlib. The entry point run_all.py sequentially reads, cleans, engineers features, selects and fits models, conducts the imbalance study, computes explanations, runs clustering and reliability diagnostics, and saves metrics and figures. A README documents dependencies and commands; nine automated tests cover foundational leakage properties and the upset-label definition.')
para('Limitations include season-level rather than game-by-game retraining, no player-level injury or lineup data, correlated features that divide importance across variables, limited permutation sampling, and a hand-chosen Elo favorite threshold and number of clusters. Results could change with alternate dataset versions or settings. Because every experiment is observational and designed for interpretation, its results should not be used to infer that a predictor causes a team to win.')
para(f'In conclusion, legitimate pregame features support test accuracy around {100*cls.loc[selected,"accuracy"]:.1f}% for the validation-selected model, above the home-team baseline. Upset prediction exhibits a clear minority-class tradeoff; Elo and recent form provide interpretable signal; clustering adds little measurable value. The strongest contribution is a reproducible, leakage-conscious evaluation that compares simple predictions with more substantive data-mining analyses.')
story.append(PageBreak())
head('References (verify full metadata before final submission)')
references=[
'Breiman, L. (2001). Random forests. Machine Learning, 45, 5-32.',
'Bunker, R., &amp; Susnjak, T. (2022). The application of machine learning techniques for predicting match results in team sport: A review. Journal of Artificial Intelligence Research, 73.',
'Chawla, N. V., Bowyer, K. W., Hall, L. O., &amp; Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357.',
'Chen, W.-J., Jhou, M.-J., Lee, T.-S., &amp; Lu, C.-J. (2021). Hybrid basketball game outcome prediction model by integrating data mining methods for the NBA. Entropy, 23(4), 477.',
'Loeffelholz, B., Bednar, E., &amp; Bauer, K. W. (2009). Predicting NBA games using neural networks. Journal of Quantitative Analysis in Sports, 5(1).',
'Lundberg, S. M., &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.',
'Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). Why should I trust you? Explaining the predictions of any classifier. ACM KDD.',
'Thabtah, F., Zhang, L., &amp; Abdelhamid, N. (2019). NBA game result prediction using feature analysis and machine learning. Annals of Data Science, 6, 103-116.',
'Lauga, N. NBA games data [Data set]. Kaggle. https://www.kaggle.com/datasets/nathanlauga/nba-games',
]
for r in references:para(r,'SmallT')

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey)
    canvas.drawRightString(7.8*inch,.43*inch,f'FSU CIS 4930  |  page {doc.page}')
    canvas.restoreState()

doc=SimpleDocTemplate(str(pdf),pagesize=(8.5*inch,11*inch),leftMargin=.75*inch,
                      rightMargin=.75*inch,topMargin=.67*inch,bottomMargin=.65*inch)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Report draft  ->',pdf)
