from langchain_llm7 import ChatLLM7
from langchain_core.messages import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import json
import re
import pandas as pd
import supabase
from pprint import pprint
from supabase import create_client, Client
import supabase
from dotenv import load_dotenv
import os
import string
load_dotenv()

supabase_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFub3Rlc3ppem92dmtuaWlqZHdwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgyNTMxMjYsImV4cCI6MjA3MzgyOTEyNn0.8fH-5neMiN8kRZSAfW_9DZ4O5wSX1e1C_KaLyo3DL1o"
supabase_url: str = "https://qnoteszizovvkniijdwp.supabase.co"
print(supabase_key )
supabase: Client = create_client(supabase_url, supabase_key)
df = pd.read_csv("chunk_2.csv")
df['volltext'] = df['volltext'].astype(str)

# Create a regex pattern: match all punctuation except /
punctuation_to_remove = f"[{re.escape(string.punctuation)}]"

# Apply replacement
df['volltext'] = df['volltext'].str.replace(punctuation_to_remove, '', regex=True)
df['volltext'] = df['volltext'].str.replace(r"\\", "", regex=True) 
df['volltext'] = df['volltext'].str.replace(r"\\[ntrbfv]", " ", regex=True)
df['volltext'] = df['volltext'].str.replace(r"\s+", " ", regex=True).str.strip()



def extract_json_block(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"No JSON object found in: {text[:200]}...")

def parse_llm_response(content: str):
    # Remove Markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
    
    # Now parse the JSON
    return json.loads(cleaned)

# -----------------------------
# Define Pydantic schema
# -----------------------------
class BookMetadata(BaseModel):
    
    title: Optional[str] = Field(None, description="The title of the book")
    authors: Optional[List[str]] = Field(None, description="List of authors")
    publisher: Optional[str] = Field(None, description="Publisher name")
    publication_year: Optional[str] = Field(None, description="Year of publication")
    publication_place: Optional[str] = Field(None, description="Place of publication")
    pages: Optional[str] = Field(None, description="Number of pages")
    format: Optional[str] = Field(None, description="Book format: gebunden, Paperback, eBook, etc.")
    remarks: Optional[str] = Field(None, description="Any remarks or notes")
    edition: Optional[str] = Field(None, description="Edition number")
    volumes: Optional[str] = Field(None, description="Number of volumes")

# -----------------------------
# Initialize LLM7 client
# -----------------------------
llm = ChatLLM7(
    model="gpt-o4-mini-2025-04-16",
    api_key='HXedu14+ImcnHY+OJWKVRwC3ZEQRDNtHUOY1PMuoSrCS/d6AP5M6qyWEoyE6/rkDxDBzZxKoAObT6OdTRQFZ9rbIdsOG02If84H2qJ041a/b80OtCwGAIr/X7Z+CGJtYPyPB23gz',
    temperature=0.0,
    max_tokens=500,
    stop=None
)
parser = PydanticOutputParser(pydantic_object=BookMetadata)

# -----------------------------
# Metadata extractor
# -----------------------------
def extract_metadata_with_schema(text: str) -> BookMetadata:
    """Extract bibliographic metadata using LLM7 + enforced schema"""

    prompt = f"""
You are a precise bibliographic metadata extractor.
You first need to think, analyze the TEXT TO ANALYZE and then extract the relevant metadata fields.
If any text is cut of, you must do your best to infer the missing information. Use your own knowledge.
You must differentiate between the different fields and return them in a JSON object.
You must follow the INSTRUCTIONS exactly.
You must know the difference between title and remarks
INSTRUCTIONS (must follow exactly):
- Analyze the TEXT TO ANALYZE and return ONLY one JSON object (no Markdown, no explanation, no extra text).
- The JSON object MUST contain exactly these keys (lowercase, underscore):
  signature, title, authors, publisher, publication_year, publication_place, pages, format, remarks, edition, volumes
- If a field is not present or cannot be determined, return null for that field.
- All non-null values MUST be strings. Numbers or years must be strings (e.g. "1926").
- The authors field MUST be either null or a JSON array of strings (e.g. ["Max Mustermann"]).
- Do NOT include any keys other than the ones listed above.
- You may reason step-by-step internally to arrive at the values, but DO NOT output your internal reasoning or any commentary — output only the single JSON object.
- If there are multiple authors, include all of them in the authors array.  
- The title field MUST contain only the title of the book, excluding authors, publisher, year, place, pages, format, remarks, edition, volumes.
- The edition field MUST contain only the edition number or description (e.g. "2nd edition", "Revised edition"), excluding any other text.
- The volumes field MUST contain only the number of volumes (e.g. "3 volumes"), excluding any other text.
### Examples

Example 1
TEXT TO ANALYZE:
83 826 Romano L Bauerneuerung Ablaufplanung vom Projekt zur Ausführung Arbeitsgruppe Andreas Bouvard Alfred Fränli u.a. Bern Bundesamt für Konjunkturfragen 1993 30cm 85 S
OUTPUT:
{{
  "title": "Bauerneuerung Ablaufplanung vom Projekt zur Ausführung Arbeitsgruppe Andreas Bouvard Alfred Fränli u.a. Bern",
  "authors": ["Romano L"],
  "publisher": "Bundesamt für Konjunkturfragen",
  "publication_year": "1993",
  "publication_place": "Bern",
  "pages": "85 S",
  "format": "30cm",
  "remarks": "",
  "edition": "",
  "volumes": ""
}}

Example 2
TEXT TO ANALYZE:
83 827 Romano L Feindiagnose im Hochbau Arbeitsgruppe Herbert Hedi ger Bruno Drr u.a. Impulsprogramm IP Bau Bern Bundesamt für Konjunkturfragen 1993 30cm 325 S
OUTPUT:
{{
  "title": "Feindiagnose im Hochbau Arbeitsgruppe Herbert Hedi ger Bruno Drr u.a. Impulsprogramm IP Bau Bern",
  "authors": ["Romano L"],
  "publisher": "Bundesamt für Konjunkturfragen",
  "publication_year": "1993",
  "publication_place": "Bern",
  "pages": "325 S",
  "format": "30cm",
  "remarks": "",
  "edition": "",
  "volumes": ""
}}

Example 3
TEXT TO ANALYZE:
83 828 Romano L Massaufnahme Aufnahmetechniken Randbedingungen Kalkulationsgrundlagen Sachbearbeitung Heinz Hirt Ralf Ammann Impulsprogramm IP Bau Bern Bundesamt für Konjunkturfragen 1992 30cm 63 S
OUTPUT:
{{
  "title": "Massaufnahme Aufnahmetechniken Randbedingungen Kalkulationsgrundlagen Sachbearbeitung Heinz Hirt Ralf Ammann",
  "authors": ["Romano L"],
  "publisher": "Bundesamt für Konjunkturfragen",
  "publication_year": "1992",
  "publication_place": "Bern",
  "pages": "63 S",
  "format": "30cm",
  "remarks": "",
  "edition": "",
  "volumes": ""
}}

Example 4
TEXT TO ANALYZE:
8 829 Magell IN Eicker Stefan Dictionary Konzepte zur Verwaltung der betrieblichen Metadaten Studien zur Wirtschaftsinformatik 7 Berlin Walter de Gruyter 1994 24cm 49 S
OUTPUT:
{{
  "title": "Dictionary Konzepte zur Verwaltung der betrieblichen Metadaten Studien zur Wirtschaftsinformatik 7",
  "authors": ["Magell IN", "Eicker Stefan"],
  "publisher": "Walter de Gruyter",
  "publication_year": "1994",
  "publication_place": "Berlin",
  "pages": "49 S",
  "format": "24cm",
  "remarks": "",
  "edition": "7",
  "volumes": ""
}}

Example 5
TEXT TO ANALYZE:
83 831 Coplien James O. Advanced C programming styles and idioms Seprinted with corrections Reading MA Addison Wesley Publishing Company 1994 24cm XIX 520 p
OUTPUT:
{{

  "title": "Advanced C programming styles and idioms",
  "authors": ["Coplien James O."],
  "publisher": "Addison Wesley Publishing Company",
  "publication_year": "1994",
  "publication_place": "Reading, MA",
  "pages": "XIX 520 p",
  "format": "24cm",
  "remarks": "Seprinted with corrections",
  "edition": "",
  "volumes": ""
}}
    {text}
    
    Return only a JSON object that strictly matches this schema:
{parser.get_format_instructions()}

    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        json_block = extract_json_block(response.content)
        parsed_output = parser.parse(json_block)
        return {
          
          "title": parsed_output.title,
          "authors": parsed_output.authors,
          "publisher": parsed_output.publisher,
          "publication_year": parsed_output.publication_year,
          "publication_place": parsed_output.publication_place,
          "pages": parsed_output.pages,
          "format": parsed_output.format,
          "remarks": parsed_output.remarks,
          "edition": parsed_output.edition,
          "volumes": parsed_output.volumes
          
        }
    except Exception as e:
        raise ValueError(f"Failed to parse response: {response.content}\nError: {e}")

import math

def clean_nans(data_list):
    cleaned = []
    for item in data_list:
        fixed = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in item.items()}
        cleaned.append(fixed)
    return cleaned

data_list = []
import math
def safe_dict(d):
    return {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in d.items()}

from time import sleep
for index, column in df.iterrows():
    print(f"Processing row {index}")
    try:
      
      output = {
        "titel_autor": column['titel_autor'],
        "rasignatur": column['rasignatur'],
        "volltext": column['volltext'],
        "file":"chunk_2",
        "field_id": index,
        "unique_key": column['unique_key']
    }
      metadata = extract_metadata_with_schema(column['volltext'])
      output.update(metadata)
      pprint(output)
      data_list.append(output)
      sleep(5)
    except Exception as e:
      print(f"Error processing row {index}: {e}")
      error_output = safe_dict({
        "Titel_Autor": column['titel_autor'],
        "RASignatur": column['rasignatur'],
        "Volltext": column['volltext'],
        "file":"chunk_2",
        "field_id": index,
        "Unique_Key": column['unique_key']
      })
      response = supabase.from_("cleaning").insert(error_output).execute()
      continue
    
    if index % 10 == 0 and index > 0:
      try:
        cleaned_data = clean_nans(data_list)
        response = supabase.from_("auth_alle").upsert(cleaned_data).execute()
        print(f"Upserted batch at index {index}, response: {response}")
        
      except Exception as e:
        print(f"Error during upsert at index {index}: {e}")
        continue
      finally:
        data_list = []  # Always reset after attempt
        sleep(2)
