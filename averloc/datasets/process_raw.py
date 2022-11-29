import os
import sys

input_dir = sys.argv[1]
final_str = ''

with open('result.jsonl', 'w') as resfp:
	for project in os.listdir(input_dir):
		proj_path = os.path.join(input_dir, project)
		for f in os.listdir(proj_path):
			print('Processing %s' % f)
			f_path = os.path.join(proj_path, f)
			with open(f_path, 'r') as fp:
				content = fp.readlines()
				content_str = ''.join(content).replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('"', '\\"')
				final_str = '{"granularity": "file", "language": "java", "code": "%s"}\n' % content_str
				resfp.write(final_str)