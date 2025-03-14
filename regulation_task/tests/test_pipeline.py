import json
from regulation_task.document_store.embedder import Embedder, VectorDatabase
from regulation_task.utils.prompt_utils import Regulatory_API_Prompt, extract_clauses_from_docx

def test_pipeline():
    """
    Test the pipeline.
    """
    embedder = Embedder()
    db = VectorDatabase(embedder)

    regulations = ["don't pet the dog after 4 pm", 
            "don't pet the dog after 5 pm", 
            "don't pet the dog after 6 pm",
            "dont pet the cat after 4 pm"]
    
    sop = ["I will pet the cat after 4 pm and the dog after 5 pm"]

    identity = open("./regulation_task/data/prompts/compliance_expert_identity.txt", "r").read()
    sop_content = open("./regulation_task/data/prompts/chunking_sop.txt", "r").read()
    regulation_content = open("./regulation_task/data/prompts/chunking_regulations.txt", "r").read()

    sop_prompter = Regulatory_API_Prompt(
        sys_prompt=identity + sop_content,
        content=sop_content,
        max_tokens=2000,
    )

    regulation_prompter = Regulatory_API_Prompt(
        sys_prompt=identity + regulation_content,
        content=regulation_content,
        max_tokens=2000,
    )

    print("Extracting regulations...")
    reg_claus_jsons = []
    for reg in regulations:
        reg_claus_str = extract_clauses_from_docx(regulation_prompter, reg)    
        regulation_prompter.reset_content()
        reg_claus_jsons.append(json.loads(reg_claus_str)['clauses'][0])      # add clause dict to list
    
    print("Adding regulations to database...")
    count = 0
    for reg_clause_json in reg_claus_jsons:
        reg_clause_str = db.convert_to_str(reg_clause_json)        # convert clause dict to formatted string
        db.add_document(reg_clause_str, count)                     # add to database
        count += 1
    
    print("Extracting SOP clauses...")
    sop_clauses = extract_clauses_from_docx(sop_prompter, sop[0])      # retrieve relevant SOP clauses
    sop_clauses = json.loads(sop_clauses)["clauses"]

    print("Getting relevant clauses...")
    for sop_clause in sop_clauses:
        sop_clause_str = db.convert_to_str(sop_clause)
        results = db.get_relevant_clauses(sop_clause_str, k=2)

        print(sop_clause_str)
        for result in results:
            print(result)

        assert len(results) == 2
    
    
test_pipeline()

    



    
        
        
            
