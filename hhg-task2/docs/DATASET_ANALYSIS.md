# Dataset Analysis — ai4bharat/MSMARCO-XI

**Total Records:** 10,000

**Columns:** source_lang, target_lang, meta, Answer, query_id, query_type, passages, Eng_Query, Eng_Answer, query


## Column Types

| Column | Type | Non-Null | Unique |
|---|---|---|---|
| source_lang | str | 10,000 | 1 |
| target_lang | str | 10,000 | 1 |
| meta | object | 10,000 | N/A |
| Answer | str | 10,000 | 6507 |
| query_id | int64 | 10,000 | 9999 |
| query_type | str | 10,000 | 5 |
| passages | object | 10,000 | N/A |
| Eng_Query | str | 10,000 | 9999 |
| Eng_Answer | str | 10,000 | 6510 |
| query | str | 10,000 | 9994 |

## Text Length Statistics

### `meta`

| Metric | Characters |
|---|---|
| Mean | 6 |
| Median (P50) | 6 |
| P75 | 6 |
| P90 | 6 |
| P95 | 6 |
| Max | 6 |
| Min | 6 |

### `passages`

| Metric | Characters |
|---|---|
| Mean | 3 |
| Median (P50) | 3 |
| P75 | 3 |
| P90 | 3 |
| P95 | 3 |
| Max | 3 |
| Min | 3 |


## Sample Records (First 5)

### Record 1

- **source_lang:** eng_Latn
- **target_lang:** hin_Deva
- **meta:** {'frequency_penalty': 0, 'max_tokens': 4096, 'model_name': 'ckpt-3epochs-sft-then-400k-kd', 'presence_penalty': 0, 'temperature': 0, 'top_p': 1}
- **Answer:** निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है और कानून में इस प्रकार से मान्यता प्राप्त होती है।
- **query_id:** 1102432
- **query_type:** DESCRIPTION
- **passages:** {'English_passages': array(['A company is incorporated in a specific nation, often within the bounds of a smaller subset of that nation, such as a state or province. The corporation is then governed b
- **Eng_Query:** . what is a corporation?
- **Eng_Answer:** A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.
- **query:** कॉर्पोरेशन क्या है?

### Record 2

- **source_lang:** eng_Latn
- **target_lang:** hin_Deva
- **meta:** {'frequency_penalty': 0, 'max_tokens': 4096, 'model_name': 'ckpt-3epochs-sft-then-400k-kd', 'presence_penalty': 0, 'temperature': 0, 'top_p': 1}
- **Answer:** रेचल कार्सन ने लिखा है कि "द ओब्लिगेशन टू एंड्योर" क्योंकि उनका मानना है कि जैसे-जैसे आदमी अवांछित कीड़ों और खरपतवारों को खत्म करने की कोशिश करता है, वैसे-वैसे वह वास्तव में पर्यावरण को प्रदूषित करके 
- **query_id:** 1102431
- **query_type:** DESCRIPTION
- **passages:** {'English_passages': array(["Read to write - grow a love of books (even if it's a comic book) Organizing an argument into paragraphs, words [that] worked together for a common purpose Walker: Using mo
- **Eng_Query:** why did rachel carson write an obligation to endure
- **Eng_Answer:** Rachel Carson writes The Obligation to Endure because believes that as man tries to eliminate unwanted insects and weeds, however he is actually causing more problems by polluting the environment.
- **query:** रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा

### Record 3

- **source_lang:** eng_Latn
- **target_lang:** hin_Deva
- **meta:** {'frequency_penalty': 0, 'max_tokens': 4096, 'model_name': 'ckpt-3epochs-sft-then-400k-kd', 'presence_penalty': 0, 'temperature': 0, 'top_p': 1}
- **Answer:** कोई उत्तर नहीं मिला।
- **query_id:** 90836
- **query_type:** ENTITY
- **passages:** {'English_passages': array(['Low Sodium Low Potassium Foods List. On this page we offer a searchable collection of nutritional data on thousands of foods for healthy diet. Healthy eating is not only b
- **Eng_Query:** chart for foods low in potassium.
- **Eng_Answer:** No Answer Present.
- **query:** पोटेशियम में कम खाद्य पदार्थों का चार्ट।

### Record 4

- **source_lang:** eng_Latn
- **target_lang:** hin_Deva
- **meta:** {'frequency_penalty': 0, 'max_tokens': 4096, 'model_name': 'ckpt-3epochs-sft-then-400k-kd', 'presence_penalty': 0, 'temperature': 0, 'top_p': 1}
- **Answer:** कोई उत्तर नहीं मिला।
- **query_id:** 55665
- **query_type:** DESCRIPTION
- **passages:** {'English_passages': array(['Take advantage of great opportunities that regularly present themselves through Shipned. Besides flat bottom tankers you will find anything from river barges to ferries, t
- **Eng_Query:** bottom front of a cargo ship
- **Eng_Answer:** No Answer Present.
- **query:** मालवाहक जहाज़ के नीचे की तरफ

### Record 5

- **source_lang:** eng_Latn
- **target_lang:** hin_Deva
- **meta:** {'frequency_penalty': 0, 'max_tokens': 4096, 'model_name': 'ckpt-3epochs-sft-then-400k-kd', 'presence_penalty': 0, 'temperature': 0, 'top_p': 1}
- **Answer:** ईमानदारी: ईमानदार होने की स्थिति। 
निष्ठा: ईमानदारी के संबंध में या उसके अतिरिक्त व्यक्ति का मूल्य और नैतिकता।
- **query_id:** 205107
- **query_type:** DESCRIPTION
- **passages:** {'English_passages': array(['Integrity is about conduct; honesty is about adherence to the facts. The person who without fail submits his timesheet every week, seeking clarification if unsure how to c
- **Eng_Query:** honesty or integrity definition
- **Eng_Answer:** Honesty: The condition of being honest. 
Integrity: The value and morals of a individual in relation or in addition to honesty.
- **query:** ईमानदारी या सच्चाई की परिभाषा


## Recommended Chunk Sizes

Based on passage length distribution:

**For `meta`:**
- If most passages are short (median=6 chars): chunk_size=256-512
- If passages vary widely (P90=6 chars): chunk_size=512-768 with 20% overlap
- Overlap recommendation: 50 chars

**For `passages`:**
- If most passages are short (median=3 chars): chunk_size=256-512
- If passages vary widely (P90=3 chars): chunk_size=512-768 with 20% overlap
- Overlap recommendation: 50 chars
