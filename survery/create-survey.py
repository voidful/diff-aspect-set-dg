import os.path

import nlp2

for idx, row in enumerate(nlp2.read_csv('./8.pt_dataset_textcsv_mode_greedy_filtersim_False_predicted.csv')[1:]):
    passage, question, answer = row[0].split("</s>")
    md = row[1]
    hd = row[2]
    html_template = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 16px;
                line-height: 1.5;
                color: #333;
                background-color: #f7f7f7;
            }}

            article {{
                margin: 0 auto;
                max-width: 800px;
                padding: 20px;
                background-color: #fff;
            }}

            p {{
                font-size: 16px;
                margin-bottom: 20px;
                word-wrap: break-word;
                white-space: pre-line;
            }}
        </style>
    </head>
    <body>
    <article>
        <p>
        {passage}
        </p>
        {question}
        </br></br>
        {answer}
    </article>
    </body>
    </html>
    """
    json_tamplate = {
        "title": "Pick the distractors set you think its better",
        "logoPosition": "right",
        "pages": [
            {
                "name": "page1",
                "elements": [
                    {
                        "type": "html",
                        "name": "question1",
                        "state": "expanded",
                        "html": html_template
                    },
                    {
                        "type": "radiogroup",
                        "name": "Select the distractor set you think it is appropriate",
                        "choices": [
                            {
                                "value": "Distractor Set 1",
                                "text": md
                            },
                            {
                                "value": "Distractor Set 2",
                                "text": hd
                            }
                        ]
                    }
                ]
            }
        ],
        "widthMode": "responsive"
    }
    outdir = nlp2.get_dir_with_notexist_create('./surveys')
    nlp2.write_json(json_tamplate, os.path.join(outdir, f'./{idx}.json'))
