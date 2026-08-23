from dataclasses import dataclass, field
from typing import Dict
import hashlib
import pathlib


@dataclass
class PipelineNode:
    file: str
    trial: int
    session: str
    checksum: str=None


@dataclass
class PipelineNetwork:
    resting_pipe: dict=field(default_factory=lambda: {
            'Raw': [],
            'Tsss': [],
            'ICA': [],
            'ICA Comp': [],
            'Apply': [],
            'Evoked': [],
            'Mark': 'Next'
        })
    empty_pipe: dict=field(default_factory=lambda: {
            'ICA Comp': 'resting_pipe',
            'Raw': [],
            'Tsss': [],
            'Apply': [],
            'Cov': [],
            'Mark': 'Next'
        })

    forward: dict=field(default_factory=lambda: {
            'Raw': 'resting_pipe',
            'Trans': [],
            'Bem': [],
            'ico1': [],
            'ico4': [],
            'Forward': [],
            'Mark': 'Forward'
        })

    # Inputs:
    # pipeline: the pipeline in PipelineNetwork that is used
    # data_id: the index to access the pipeline dictionary
    # trials: which trials this data accounts for. If none, it will be interpreted as
    # if all trials use the same data source, such as a Raw resting file
    # sessions: which sessions the data accounts for. Typically None for MRIs and Emptyrooms
    # The function raises an error if there is an inconsistency with the data progression
    def verify_data(self, pipeline, data_id, trial=None, session=None):
        if data_id not in pipeline: raise KeyError(f"invalid data_id passed to PipelineNetwork.{pipeline}!")

        error_list = []
        for key in pipeline:
            if key == data_id: break
            if key == 'Mark': raise KeyError(f"Was not able to find {data_id} in pipeline key iteration!")
            verify_dict = {
                "Key": key,
                "Info": [] #2D list. 0: trial, 1: session
            }
            if type(pipeline[key]) is str: node_list = self._locate_data(pipe_name=pipeline[key], data_id=key)
            else: node_list = pipeline[key]
            match_found = False
            for node in node_list:
                if trial is not None and node.trial != trial: # trials do not match
                    continue
                if session is not None and node.session != session: # sessions do not match
                    continue

                #trial and session are compatible
                match_found = True
                assert pathlib.Path(node.file).exists(), f"{node.file} was changed and/or deleted!"
                with open(node.file, "rb") as f:
                    checksum = hashlib.file_digest(f, "sha256").hexdigest()

                if node.checksum is not None and checksum == node.checksum: continue

                verify_dict['Info'].append([node.trial, node.session])

            if match_found is False:
                raise RuntimeError(f"{pipeline}: {key} | trial:{trial} session:{session} "
                    f"does not exist! please recompute {key}!")
            if len(verify_dict['Info']) == 0: continue

            error_list.append(verify_dict)

        if len(error_list) == 0: return

        network = self.__dict__
        for key in network.keys():
            if network[key] == pipeline:
                pipe_name = key

        for nodedict in error_list:
            print(f"\n\noutdated files identified in {pipe_name}: {key}")
            for elem in nodedict['Info']:
                print(f"\ttrial: {elem[0]}, session: {elem[1]}")

        raise RuntimeError(f"Verification of {data_id} in {pipe_name} failed!")
        

                        # if node.checksum is None: raise RuntimeError(f"{key} trials={node.trials} "
                        #     f"sessions={node.sessions} is outdated / does not exist! Please update"
                        #     f"this file before running this function again.")

    # Inputs:
    # pipeline: the pipeline in PipelineNetwork that is used
    # data_id: the index to access the pipeline dictionary
    # trials: which trials this data accounts for. If none, it will be interpreted as
    # if all trials use the same data source, such as a Raw resting file
    # sessions: which sessions the data accounts for. Typically None for MRIs and Emptyrooms

    def mark_nodes(self, pipeline, start_id, trial, session):
        if pipeline['Mark'] == 'Next':
            pipeline_keys = list(pipeline.keys())
            pipeline_keys = pipeline_keys[pipeline_keys.index(start_id) + 1:]
            for key in pipeline_keys:
                if key == 'Mark': return
                if type(pipeline[key]) is str: continue
                for node in pipeline[key]:
                    if node.trial == trial or trial == 'all': # trials match
                        if node.session == session or session == 'all': #sessions match
                            node.checksum = None
            return
        
        if pipeline['Mark'] == 'Forward':
            if start_id != 'Forward':
                for node in pipeline['Forward']:
                    if node.trial == trial or trial == 'all': # trials match
                        if node.session == session or session == 'all': #sessions match
                            node.checksum = None
                            return
            return
    
    def save_checksum(self, pipeline, data_id, trial, session, dst, overwrite=False):
        if data_id not in pipeline: raise KeyError(f"invalid data_id passed to PipelineNetwork!{pipeline}, {data_id}")

        dst = str(dst)

        

        # TODO: implement overwrite functionality. When doing so, make sure to save the file name 
        for node in pipeline[data_id]:
            if node.trial != trial: # trials do not match
                continue
            if node.session != session: # sessions do not match
                continue

            with open(node.file, "rb") as f:
                checksum = hashlib.file_digest(f, "sha256").hexdigest()

            if checksum == node.checksum: return 0 # Nothing was modified

            node.checksum = checksum
            # print(pipeline, data_id, trial, session)
            # print(trial, session)
            # print(len(pipeline), type(pipeline))
            # print(f"\n{pipeline}\n")
            # print(self)
            self.mark_nodes(pipeline=pipeline, start_id=data_id, trial=trial, session=session)
            return 1 # An existing node was modified

                



        node = PipelineNode(file=dst, trial=trial, session=session)  
        with open(node.file, "rb") as f:
            node.checksum = hashlib.file_digest(f, "sha256").hexdigest()

        pipeline[data_id].append(node)
        return 2 # A new Node was created


    def remove_checksum(self, pipeline, data_id, node):
        if data_id not in pipeline: raise KeyError(f"invalid data_id passed to PipelineNetwork.{pipeline}!")

        for idx in range(len(pipeline[data_id])):
            if pipeline[data_id][idx].trials == node.trials and \
            pipeline[data_id][idx].sessions == node.sessions:
                pipeline[data_id].pop(idx)

    def copy_data(self, src_pipeline, src_id, target_pipeline, target_id):
        if src_id not in src_pipeline: raise RuntimeError(f"invalid src passed to PipelineNetwork.{src_pipeline}!")
        if target_id not in target_pipeline: raise KeyError(f"invalid src passed to PipelineNetwork.{target_pipeline}!")

        for src_node in src_pipeline[src_id]:
            tgt_node = self.find_node(pipeline=target_pipeline, data_id=target_id, \
                trial=src_node.trial, session=src_node.session)
            if src_node.checksum != tgt_node.checksum: # file will be updated
                print(src_node, tgt_node)
                self.mark_nodes(pipeline=target_pipeline, start_id=target_id, 
                    trial=tgt_node.trial, session=tgt_node.session)
        target_pipeline[target_id] = src_pipeline[src_id]
        # self.mark_nodes(pipeline=target_pipeline, start_id=target_id, trial='all', session='all')

    def find_node(self, pipeline, data_id, trial, session):
        if data_id not in pipeline: raise KeyError(f"invalid data_id passed to PipelineNetwork.{pipeline}!")
        if type(pipeline[data_id]) is str: node_list = self._locate_data(pipe_name=pipeline[data_id], data_id=data_id)
        else: node_list = pipeline[data_id]

        for node in node_list:
            if node.trial == trial and node.session == session:
                return node


    def _locate_data(self, pipe_name, data_id):
        if type(pipe_name) is not str: raise ValueError
        network = self.__dict__
        keys = network.keys()
        if pipe_name not in keys: raise KeyError
        return network[pipe_name][data_id]


    
    


