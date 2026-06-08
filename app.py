import streamlit as st
import pandas as pd
import os
import csv
import string    
from datetime import datetime
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# ⚙️ CONFIG & SESSION STATE
# ===============================

# Updated Data Loading Function

```python
@st.cache_data
def load_and_prep_data():

    file_path = "knowledge_base.csv" if os.path.exists("knowledge_base.csv") else "data.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=[
            'Material Code',
            'Material Description',
            'GL Account',
            'Valuation Class'
        ])

    required_cols = [
        'Material Code',
        'Material Description',
        'GL Account',
        'Valuation Class'
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    # Search profile
    df['profile'] = (
        df['Material Code'].fillna('').astype(str) + ' ' +
        df['Material Description'].fillna('').astype(str) + ' ' +
        df['GL Account'].fillna('').astype(str) + ' ' +
        df['Valuation Class'].fillna('').astype(str)
    )

    return df
```

# Updated Search Function

```python
def get_combined_matches(query, dataframe, top_n=5):

    if dataframe.empty:
        return []

    choices = dataframe['profile'].tolist()

    fuzzy_results = process.extract(
        query,
        choices,
        scorer=fuzz.token_set_ratio,
        limit=top_n
    )

    results = []

    for match in fuzzy_results:

        score = match[1]
        idx = match[2] if len(match) > 2 else dataframe.index[
            dataframe['profile'] == match[0]
        ][0]

        if score > 70:

            results.append({
                "score": score / 100,
                "material_code": dataframe.iloc[idx]['Material Code'],
                "material_desc": dataframe.iloc[idx]['Material Description'],
                "gl_account": dataframe.iloc[idx]['GL Account'],
                "valuation_class": dataframe.iloc[idx]['Valuation Class'],
                "idx": idx
            })

    if not results:

        vectorizer = TfidfVectorizer(stop_words='english')

        tfidf_matrix = vectorizer.fit_transform(
            dataframe['profile']
        )

        query_vec = vectorizer.transform([query])

        cosine_sim = cosine_similarity(
            query_vec,
            tfidf_matrix
        ).flatten()

        nlp_indices = cosine_sim.argsort()[-top_n:][::-1]

        for idx in nlp_indices:

            if cosine_sim[idx] > 0.10:

                results.append({
                    "score": cosine_sim[idx],
                    "material_code": dataframe.iloc[idx]['Material Code'],
                    "material_desc": dataframe.iloc[idx]['Material Description'],
                    "gl_account": dataframe.iloc[idx]['GL Account'],
                    "valuation_class": dataframe.iloc[idx]['Valuation Class'],
                    "idx": idx
                })

    return results
```

# Replace Existing Response Block

Replace:

```python
st.session_state.messages.append({
    "role": "assistant",
    "content": f"**{results[0]['q']}**\n\n{results[0]['a']}"
})
```

With:

```python
st.session_state.messages.append({
    "role": "assistant",
    "content":
        f"**Material Code:** {results[0]['material_code']}\n\n"
        f"**Material Description:** {results[0]['material_desc']}\n\n"
        f"**GL Account:** {results[0]['gl_account']}\n\n"
        f"**Valuation Class:** {results[0]['valuation_class']}"
})
```

# Replace Suggestion Buttons

Replace:

```python
st.caption("Common matches:")

for r in st.session_state.temp_results:
    if st.button(
        f"👉 {r['q']}",
        key=f"sug_btn_{r['idx']}",
        use_container_width=True
    ):
```

With:

```python
st.caption("Matching Materials:")

for r in st.session_state.temp_results:

    button_text = (
        f"👉 {r['material_code']} - "
        f"{r['material_desc']}"
    )

    if st.button(
        button_text,
        key=f"sug_btn_{r['idx']}",
        use_container_width=True
    ):
```
