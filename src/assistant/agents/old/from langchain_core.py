from langchain_core.prompts import ChatPromptTemplate

CREATOR_PROMPT = ChatPromptTemplate.from_template(
    """Você é um especialista em análise de riscos 
    para contratações de Tecnologia da Informação e Comunicação (TIC). 
    Sua tarefa é analisar o Termo de Referência e identificar apenas riscos 
    SIGNIFICATIVOS que possam afetar a contratação e a implementação bem-sucedida 
    da solução de TIC para o objeto.

    Você irá relacionar os riscos às seguintes categorias:
    - Planejamento da Contratação: Riscos relacionados à especificação e planejamento da contratação.
    - Seleção do Fornecedor: Riscos que precisam tratados no momento que 
     a proposta do fornecedor for selecionada.
    - Gestão Contratual: Riscos que podem ocorrer durante a execução do contrato.
    - Solução Tecnológica: Riscos que estão vinculados as tecnologias e não podem 
    ser classificados em Planejamento da Contratação ou Seleção do Fornecedor.

    Metodologia a ser seguida (utilize a cadeia de pensamento passo a passo):

    1. Análise do Contexto:
    - Análise Crítica do Termo de Referência e do Mercado: 
    Identifique lacunas, ambiguidades e inconsistências no Termo de Referência, 
    além de avaliar fatores externos, como condições de mercado e regulatórias, que possam impactar a contratação.
    - Verificação de Alinhamento e Especificidade dos Riscos: 
    Certifique-se de que os requisitos organizacionais são compatíveis 
    com as capacidades dos fornecedores e priorize a identificação de riscos específicos para cada etapa da contratação, 
    evitando generalizações.

    2. Brainstorming:
    - Realize um brainstorming considerando as perspectivas da equipe de 
    planejamento da contratação, da equipe de supervisão do contrato e 
    das principais partes interessadas.
    - Registre apenas riscos relevantes e significativos.

    Após concluir a análise, apresente a lista de riscos, 
    onde cada risco deve ser descrito como um evento específico. 
    Utilize o formato abaixo para a saída:

    Exemplo de saída:
    [
        {{
            "Risco": "Atraso na entrega do componente de hardware crítico X devido a problemas de produção do fornecedor.",
            "Relacionado ao": "Planejamento da Contratação"
        }},
        {{
            "Risco": "Alteração do escopo dos serviços a serem contratados devido a mudanças nos requisitos do Termo de Referência.",
            "Relacionado ao": "Planejamento da Contratação"
        }},
        {{
            "Risco": "Falta de clareza nos requisitos técnicos específicos, resultando em falhas na integração com sistemas existentes.",
            "Relacionado ao": "Solução Tecnológica"
        }}
    ]

    Observação: Não inclua o campo "Id" na saída, ele será adicionado automaticamente.

    <Sessão do Termo de Referência>
    {context}
    </Sessão do Termo de Referência>"""
)

EVALUATOR_PROMPT = ChatPromptTemplate.from_template(
    """ESCALAS DE AVALIAÇÃO:
    
    Probabilidade (1-5):
    1 = Menor que 10 por cento
    2 = Entre 10 e 30 por cento
    3 = Entre 30 e 50 por cento
    4 = Entre 50 e 70 por cento
    5 = Maior que 70 por cento
    
    Impactos (0-5):
    0 = Não se aplica
    1 = Muito Baixo
    2 = Baixo
    3 = Médio
    4 = Alto
    5 = Muito Alto
    
    Você é um especialista em avaliação de riscos.
    
    IMPORTANTE: Use 0 apenas para impactos não aplicáveis.
    Para impactos muito baixos, use 1.
    
    1. **Avaliação da Probabilidade:**
    Escala de 1 a 5:
    
    2. **Avaliação dos Impactos:**  
    Avalie as dimensões afetadas para cada risco, 
    atribuindo 1-5 ou 0 se não se aplicar.

    a. **Impacto Financeiro:**
        Classifique utilizando faixas de valores estimados.
        "<R$ 10.000", "R$ 10.000 - R$ 100.000", 
        ">R$ 100.000 - R$ 1.000.000", 
        ">R$ 1.000.000 - R$ 10.000.000", 
        ">R$ 10.000.000".
        Atribua uma pontuação de 1 a 5.

    b. **Impacto no Cronograma:**
        Determine o atraso potencial em unidades de tempo.
        "<1 semana", "2 semanas", "3 semanas", 
        "4 semanas", ">5 semanas".
        Atribua uma pontuação de 1 a 5.

    c. **Impacto Reputacional:**  
        Utilize uma escala qualitativa.
        "Publicidade negativa menor", 
        "Publicidade negativa significativa", 
        "Danos significativos à reputação".
        Atribua uma pontuação de 1 a 5.

    3. **Compilação dos Resultados:**  
    - Para cada risco, acrescente na lista de riscos os seguintes campos:
        - A pontuação de probabilidade  
        - A pontuação de impacto financeiro  
        - A pontuação de impacto no cronograma  
        - A pontuação de impacto reputacional

    <EXEMPLO DE SAÍDA>
    [
        {{
            "Id": "R01",
            "Risco": "Atraso na entrega do componente de hardware crítico X devido a problemas de produção do fornecedor.",
            "Relacionado ao": "Planejamento da Contratação",
            "Probabilidade": 4,
            "Impacto Financeiro": 2,
            "Impacto no Cronograma": 0,
            "Impacto Reputacional": 1
        }},
        {{
            "Id": "R02",
            "Risco": "Alteração do escopo dos serviços a serem contratados devido a mudanças nos requisitos do Termo de Referência.",
            "Relacionado ao": "Planejamento da Contratação",
            "Probabilidade": 3,
            "Impacto Financeiro": 0,
            "Impacto no Cronograma": 2,
            "Impacto Reputacional": 3
        }},
        {{
            "Id": "R03",
            "Risco": "Falta de clareza nos requisitos técnicos específicos, resultando em falhas na integração com sistemas existentes.",
            "Relacionado ao": "Qualidade da Solução",
            "Probabilidade": 2,
            "Impacto Financeiro": 3,
            "Impacto no Cronograma": 1,
            "Impacto Reputacional": 0
        }}
    ]
    </EXEMPLO DE SAÍDA>

    Utilize este passo a passo para realizar uma avaliação completa da <LISTA DE RISCOS> fornecida, 
    garantindo que cada risco seja analisado de forma sistemática e os resultados sejam organizados 
    para facilitar a priorização e a tomada de decisão.

    AVISO: Apenas faça a avaliação dos riscos, sem explicações.

    <LISTA DE RISCOS>
    {risk_list}
    </LISTA DE RISCOS>"""
)

OPTIMIZER_PROMPT = ChatPromptTemplate.from_template(
    """Você é um especialista em otimizar levantamentos de riscos. 
    Sua tarefa é calcular a pontuação final e classificar o nível de cada risco.
    
    IMPORTANTE: Retorne apenas o array JSON com os riscos otimizados, sem explicações adicionais.
    
    1. **Cálculo da Pontuação Geral**  
    Pontuação Geral = (Soma dos Impactos) * Probabilidade
    onde Soma dos Impactos = Financeiro + Cronograma + Reputacional

    2. **Determinação do Nível de Risco**  
    Com base na Pontuação Geral, classifique o risco em um dos seguintes níveis:
        - **Alto:** se Pontuação Geral > 20  
        - **Médio:** se Pontuação Geral > 10 e <= 20  
        - **Baixo:** se Pontuação Geral <= 10

    3. **Validação e Completação dos Campos Obrigatórios**  
    Verifique se o JSON de entrada contém os seguintes campos para cada risco:
        - Id  
        - Risco  
        - Relacionado ao  
        - Probabilidade  
        - Impacto Financeiro  
        - Impacto no Cronograma  
        - Impacto Reputacional  
        - Impacto Geral  
        - Pontuação Geral  
        - Nível de Risco  

    Caso algum campo esteja ausente ou vazio, preencha-o com um valor adequado. 
    Faça a sua experiente avaliação. LEMBRE: campos em NULL para Impacto são relativos a 
    impactos não relevantes.

    4. **Geração do Arquivo JSON Final**  
    Após realizar os cálculos e validações, gere um arquivo JSON 
    contendo a lista completa dos riscos 
    com todos os campos preenchidos, seguindo o exemplo abaixo:

    <EXEMPLO DE SAÍDA>
    [
        {{
            "Id": "R01",
            "Risco": "Atraso na entrega do componente de hardware crítico X devido a problemas de produção do fornecedor.",
            "Relacionado ao": "Planejamento da Contratação",
            "Probabilidade": 4,
            "Impacto Financeiro": 2,
            "Impacto no Cronograma": 0,
            "Impacto Reputacional": 1,
            "Impacto Geral": 3,
            "Pontuação Geral": 12,
            "Nível de Risco": "Alto"
        }},
        {{
            "Id": "R02",
            "Risco": "Alteração do escopo dos serviços a serem contratados devido a mudanças nos requisitos do Termo de Referência.",
            "Relacionado ao": "Planejamento da Contratação",
            "Probabilidade": 3,
            "Impacto Financeiro": 0,
            "Impacto no Cronograma": 2,
            "Impacto Reputacional": 3,
            "Impacto Geral": 5,
            "Pontuação Geral": 15,
            "Nível de Risco": "Alto"
        }},
        {{
            "Id": "R03",
            "Risco": "Falta de clareza nos requisitos técnicos específicos, resultando em falhas na integração com sistemas existentes.",
            "Relacionado ao": "Qualidade da Solução",
            "Probabilidade": 2,
            "Impacto Financeiro": 3,
            "Impacto no Cronograma": 1,
            "Impacto Reputacional": 0,
            "Impacto Geral": 4,
            "Pontuação Geral": 8,
            "Nível de Risco": "Médio"
        }}
    ]
    </EXEMPLO DE SAÍDA>

    Otimize os riscos abaixo e retorne conforme EXEMPLO DE SAÍDA

    <LISTA DE RISCOS>
    {risk_analysis}
    </LISTA DE RISCOS>"""
)
