import json
import pandas as pd
from pathlib import Path
from ..utils import process_risk_data

def verify_risk_processing():
    """Test risk data processing without model evaluation"""
    try:
        # Load sample data directly
        data_path = Path(__file__).parent / "output_risk_eval.json"
        
        with open(data_path, 'r', encoding='utf-8') as f:
            sample_risks = json.load(f)
            
        # Convert to JSON string to simulate real input
        json_input = json.dumps(sample_risks, ensure_ascii=False)
        
        # Process data
        df = process_risk_data(json_input)
        
        # Basic validation
        required_columns = [
            'Id', 'Risco', 'Relacionado ao', 'Probabilidade',
            'Impacto Financeiro', 'Impacto no Cronograma', 
            'Impacto Reputacional', 'Impacto Geral',
            'Pontuação Geral', 'Nível de Risco'
        ]
        
        # Check structure
        assert all(col in df.columns for col in required_columns), "Missing columns"
        assert len(df) == 78, "Incorrect number of risks processed"
        
        # Check specific known values from output_risk_eval.json
        sample_risk = df[df['Id'] == 'R013'].iloc[0]
        assert sample_risk['Impacto Geral'] == 11, "R013 Impacto Geral calculation error"
        assert sample_risk['Pontuação Geral'] == 44, "R013 Pontuação Geral calculation error"
        assert sample_risk['Nível de Risco'] == 'Alto', "R013 classification error"
        
        sample_risk = df[df['Id'] == 'R001'].iloc[0]
        assert sample_risk['Nível de Risco'] == 'Médio', "R001 classification error"
        
        print("All validations passed!")
        print("\nSample processed data:")
        print(df[['Id', 'Pontuação Geral', 'Nível de Risco']].head(10))
        
        # Export to Excel for manual inspection
        output_path = Path(__file__).parent / "processed_risks.xlsx"
        df.to_excel(output_path, index=False)
        print(f"\nFull results exported to: {output_path}")

    except Exception as e:
        print(f"Verification failed: {str(e)}")
        raise

if __name__ == "__main__":
    verify_risk_processing() 