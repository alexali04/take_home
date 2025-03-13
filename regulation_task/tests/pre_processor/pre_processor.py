from regulation_task.pre_processor.pre_processing import SOP_Processor

def test_sop_processor():
    sop_processor = SOP_Processor("./regulation_task/data/sop/original.docx", cut_off=True, cut_off_length=100)
    paras = sop_processor.manual_para_chunking()

    for para in paras:
        assert len(para) <= 100
    
