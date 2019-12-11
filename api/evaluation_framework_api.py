import flask    
from flask import request
from flask import jsonify
from flask import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.manager import FrameworkManager

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/run', methods=['GET'])
def run_evaluation_framework():

	query_parameters = request.args
	input_file = query_parameters.get('vector_filename')

	evaluation_manager = FrameworkManager()
	scores = evaluation_manager.evaluate(input_file)

	html = ""
	html += scores['Classification'].to_html() + "<br>"
	html += scores['Regression'].to_html() + "<br>"

	return html

@app.route('/classification/run', methods=['GET'])
def run_classification():
	task = 'Classification'

	query_parameters = request.args
	input_file = query_parameters.get('vector_filename')

	evaluation_manager = FrameworkManager()
	scores = evaluation_manager.evaluate(input_file, tasks=[task])
	
	return scores[task].to_html()

@app.route('/regression/run', methods=['GET'])
def run_regression():
	task = 'Regression'

	query_parameters = request.args
	input_file = query_parameters.get('vector_filename')

	evaluation_manager = FrameworkManager()
	scores = evaluation_manager.evaluate(input_file, tasks=[task])
	
	return scores[task].to_html()


@app.route('/clustering/run', methods=['GET'])
def run_clustering():
	task = 'Clustering'

	query_parameters = request.args
	input_file = query_parameters.get('vector_filename')

	evaluation_manager = FrameworkManager()
	scores = evaluation_manager.evaluate(input_file, tasks=[task])
	
	print scores


	return scores[task].to_html()

@app.route('/document_similarity/run', methods=['GET'])
def run_document_similarity():
	task = 'DocumentSimilarity'

	query_parameters = request.args
	input_file = query_parameters.get('vector_filename')

	evaluation_manager = FrameworkManager()
	scores = evaluation_manager.evaluate(input_file, tasks=[task])
	
	return scores[task].to_html()

@app.route('/entity_relatedness/run', methods=['GET'])
def run_entity_relatedness():
	task = 'EntityRelatedness'

	query_parameters = request.args
	input_file = query_parameters.get('vector_filename')

	evaluation_manager = FrameworkManager()
	scores = evaluation_manager.evaluate(input_file, tasks=[task])
	
	print scores

	return scores[task].to_html()

@app.route('/semantic_analogies/run', methods=['GET'])
def run_semantic_analogies():
	task = 'SemanticAnalogies'

	query_parameters = request.args
	input_file = query_parameters.get('vector_filename')


	evaluation_manager = FrameworkManager()
	scores = evaluation_manager.evaluate(input_file, tasks=[task])
	
	print scores


	return scores[task].to_html()

app.run()
