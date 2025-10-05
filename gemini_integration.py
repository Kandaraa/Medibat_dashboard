import google.generativeai as genai

import json
import os
import re
import pandas as pd


# Gemini API Configuration
def setup_gemini_api(api_key):
    """Configure the Gemini API with the provided API key"""
    genai.configure(api_key=api_key)
    return True

# Function to prepare data to send to Gemini
def prepare_data_for_gemini(idx_stats, hours_stats, conf_stats, maint_stats, cause_stats, categories_stats, vid_stats):
    """Prepares dashboard data for sending to the Gemini API"""
    data = {
        "equipment_status": {
            "total": idx_stats["total"],
            "out_of_service": idx_stats["panne"],
            "functional": idx_stats["marche"],
            "to_be_checked": idx_stats["verifier"],
            "pct_out_of_service": round((idx_stats["panne"] / idx_stats["total"]) * 100 if idx_stats["total"] > 0 else 0, 1)
        },
        "preventive_maintenance": {
            "global_annual_average": round(hours_stats["global_avg"], 3) if not pd.isna(hours_stats["global_avg"]) else None,
            "by_category": hours_stats["by_cat"].to_dict("records") if isinstance(hours_stats["by_cat"], pd.DataFrame) else []
        },
        "fluids_conformity": {
            "pct_compliant": round(conf_stats["pct_conf"], 1),
            "pct_partial": round(conf_stats["pct_partielle"], 1),
            "total_count": conf_stats["total"],
            "compliant_count": conf_stats["conf_count"],
            "partial_count": conf_stats["part_count"]
        },
        "maintenance": {
            "misplanning_count": maint_stats["n_misplan"],
            "avg_downtime_days": round(maint_stats["avg_duration_days"], 1) if not pd.isna(maint_stats["avg_duration_days"]) else None,
            "downtime_stats": {
                "min": round(maint_stats["valid_durations"].min(), 1) if len(maint_stats["valid_durations"]) > 0 else None,
                "max": round(maint_stats["valid_durations"].max(), 1) if len(maint_stats["valid_durations"]) > 0 else None,
                "median": round(maint_stats["valid_durations"].median(), 1) if len(maint_stats["valid_durations"]) > 0 else None,
                "std_dev": round(maint_stats["valid_durations"].std(), 1) if len(maint_stats["valid_durations"]) > 0 else None
            },
            #"top_longest_downtimes": []
        },
        "root_causes": {
            "global_distribution": cause_stats["pct_tbl"].to_dict("records") if not cause_stats["pct_tbl"].empty else []
        },
        "breakdown_categories": {
            "availability": {
                "distribution": categories_stats["disponibilite"]["pct"].to_dict("records") if categories_stats["disponibilite"]["pct"] is not None and not categories_stats["disponibilite"]["pct"].empty else []
            },
            "cost": {
                "distribution": categories_stats["cout"]["pct"].to_dict("records") if categories_stats["cout"]["pct"] is not None and not categories_stats["cout"]["pct"].empty else []
            },
            "complexity": {
                "distribution": categories_stats["complexite"]["pct"].to_dict("records") if categories_stats["complexite"]["pct"] is not None and not categories_stats["complexite"]["pct"].empty else []
            }
        },
        "oil_change_schedule": {
            "on_schedule": vid_stats["respected"],
            "slightly_dangerous": vid_stats["yellow"],
            "dangerous": vid_stats["orange"],
            "extremely_dangerous": vid_stats["red"],
            "off_schedule_percentage": round(((vid_stats["yellow"] + vid_stats["orange"] + vid_stats["red"]) / 150) * 100, 1)
        }
    }

    
    return data

# Function to create the prompt for Gemini
def create_gemini_prompt(data):
    """Creates a detailed prompt for Gemini with context and data"""
    prompt = """
    # Equipment Data Analysis and Strategic Recommendations for a Construction Company

    ## Context
    You are an expert consultant in heavy equipment management for a construction and public works company. The following data comes from the company's maintenance management system and concerns their fleet of equipment (backhoe loaders, loaders, excavators, trucks, etc.). Your task is to analyze this data and provide strategic recommendations to improve operations, reduce downtime, and optimize maintenance costs.

    ## Analysis Data
    """
    
    # Add formatted data to the prompt
    prompt += "\n### Equipment Status\n"
    prompt += f"- Total equipment: {data['equipment_status']['total']}\n"
    prompt += f"- Out of service equipment: {data['equipment_status']['out_of_service']} ({data['equipment_status']['pct_out_of_service']}%)\n"
    prompt += f"- Functional equipment: {data['equipment_status']['functional']}\n"
    prompt += f"- Equipment to be checked: {data['equipment_status']['to_be_checked']}\n"
    
    prompt += "\n### Preventive Maintenance\n"
    if data['preventive_maintenance']['global_annual_average'] is not None:
        prompt += f"- Global annual average: {data['preventive_maintenance']['global_annual_average']}\n"
    
    if data['preventive_maintenance']['by_category']:
        prompt += "- By equipment category:\n"
        for cat in data['preventive_maintenance']['by_category']:
            cat_name = list(cat.values())[0] if cat else "Not specified"
            avg = cat.get('avg_per_year', 'N/A')
            prompt += f"  * {cat_name}: {avg}\n"
    
    prompt += "\n### Fluids Conformity\n"
    prompt += f"- Compliant percentage: {data['fluids_conformity']['pct_compliant']}%\n"
    prompt += f"- Partially compliant percentage: {data['fluids_conformity']['pct_partial']}%\n"
    
    prompt += "\n### Corrective Maintenance\n"
    prompt += f"- Number of misplanned interventions: {data['maintenance']['misplanning_count']}\n"
    if data['maintenance']['avg_downtime_days'] is not None:
        prompt += f"- Average downtime: {data['maintenance']['avg_downtime_days']} days\n"
    
    if data['maintenance']['downtime_stats']['min'] is not None:
        prompt += "- Downtime statistics (days):\n"
        prompt += f"  * Minimum: {data['maintenance']['downtime_stats']['min']}\n"
        prompt += f"  * Maximum: {data['maintenance']['downtime_stats']['max']}\n"
        prompt += f"  * Median: {data['maintenance']['downtime_stats']['median']}\n"
        prompt += f"  * Standard deviation: {data['maintenance']['downtime_stats']['std_dev']}\n"
    

    
    prompt += "\n### Root Causes of Breakdowns\n"
    if data['root_causes']['global_distribution']:
        prompt += "- Global distribution:\n"
        for cause in data['root_causes']['global_distribution']:
            prompt += f"  * {cause.get('cause', 'Not specified')}: {cause.get('pct', 0)}%\n"
    
    prompt += "\n### Breakdown Categories\n"
    
    # Availability
    if data['breakdown_categories']['availability']['distribution']:
        prompt += "- By parts availability:\n"
        for item in data['breakdown_categories']['availability']['distribution']:
            prompt += f"  * {item.get('classe', 'Not specified')}: {item.get('pct', 0)}%\n"
    
    # Cost
    if data['breakdown_categories']['cost']['distribution']:
        prompt += "- By repair cost:\n"
        for item in data['breakdown_categories']['cost']['distribution']:
            prompt += f"  * {item.get('classe', 'Not specified')}: {item.get('pct', 0)}%\n"
    
    # Complexity
    if data['breakdown_categories']['complexity']['distribution']:
        prompt += "- By repair complexity:\n"
        for item in data['breakdown_categories']['complexity']['distribution']:
            prompt += f"  * {item.get('classe', 'Not specified')}: {item.get('pct', 0)}%\n"
    
    prompt += "\n### Oil Change Schedule\n"
    prompt += f"- On schedule (<3 months): {data['oil_change_schedule']['on_schedule']}\n"
    prompt += f"- Slightly dangerous (3-6 months): {data['oil_change_schedule']['slightly_dangerous']}\n"
    prompt += f"- Dangerous (6-12 months): {data['oil_change_schedule']['dangerous']}\n"
    prompt += f"- Extremely dangerous (>12 months): {data['oil_change_schedule']['extremely_dangerous']}\n"
    prompt += f"- Off-schedule percentage: {data['oil_change_schedule']['off_schedule_percentage']}%\n"
    
    prompt += """
    ## Requested Tasks
    1. Analyze this data and identify critical issues affecting the company's operations.
    2. Provide a detailed analysis of trends and patterns you observe in the data.
    3. Propose concrete strategic recommendations to:
       - Reduce equipment downtime
       - Improve preventive maintenance efficiency
       - Optimize spare parts inventory management
       - Reduce maintenance costs
       - Improve oil change scheduling
    4. Suggest additional key performance indicators (KPIs) that the company should track.
    5. Propose a short-term (3 months) and medium-term (1 year) action plan with concrete steps.

    Your analysis should be structured, factual, and oriented toward concrete actions that the company can implement immediately.
    
    ## Important
    1. Do NOT start your response with phrases like "Here's a structured analysis..." or "Absolutely. Here's my analysis..."
    2. End your analysis with a "Quick Summary" section that includes bullet points of the most important findings and recommendations.
    """
    
    return prompt

# Function to call the Gemini API and get recommendations
def get_gemini_recommendations(prompt, model_name="gemini-2.5-pro"):
    """Calls the Gemini API with the prompt and returns the recommendations"""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Remove common introductory phrases
        text = response.text
        text = re.sub(r'^(Absolutely\.|Here\'s|Sure\.|Of course\.|I\'ll provide|Based on the data provided,)[^\n]*\n', '', text)
        text = re.sub(r'^[^\n]*structured analysis[^\n]*\n', '', text)
        
        return text
    except Exception as e:
        return f"Error when calling the Gemini API: {str(e)}"

# Function to format recommendations for display in Streamlit
def format_recommendations_for_display(recommendations):
    """Formats recommendations for display in Streamlit"""
    return recommendations