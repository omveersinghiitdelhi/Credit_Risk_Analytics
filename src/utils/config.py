"""
Define config class for all required information to run the template
"""

class MissingAndSVConfig:
    """ class for missing and special value config"""
    def __init__(self,config_dict):
        self.sv_indicator_list=config_dict.get('sv_indicator_list')
        self.missing_indicators=config_dict.get('missing_indicators')
        self.missing_strategy_numeric=config_dict.get('missing_strategy_numeric')
class VarSummaryConfig:
    """Calss for variable summary"""
    def __init__(self,config_dic):
        self.max_category_number = config_dic.get('max_category_number')
        self.high_missing_ratio = config_dic.get('high_missing_ratio')
class UnivariateConfig:
    """Class for Univariate Statistics config"""
    def __init__(self,config_dict):
        self.num_categories=config_dict.get('num_categories')
        self.bin=config_dict.get('bin')
        self.threshold_iv=config_dict.get('Threshold_IV')
        self.threshold_gini=config_dict.get('ThresholdGini')
        self.threshold_sp=config_dict.get('ThresholdSP')
        self.threshold_rsq=config_dict.get('ThresholdRsq')
class CollinearityConfig:
    """ Class for collinearity config """
    def __init__(self,config_dict):
        self.ci_threshold = config_dict.get('ci_threshold')
        self.var_prop_threshold=config_dict.get('var_prop_threshold')
        self.add_intercept=config_dict.get('add_intercept')
    
    @property
    def add_intercept(self):
        return self._add_intercept
    @add_intercept.setter
    def add_intercept(self,value):
        if not isinstance(value,bool):
            raise TypeError('clusteringconfig.add_intercept should be of type "bool" ') 
class ClusteringConfig:
    """ class for clustering config"""
    def __init__(self, config_dict):
        self.min_features = config_dict.get('min_features')
        self.metric = config_dict.get('metric')
        self.sample = config_dict.get('sample')
        self.autodrop = config_dict.get('autodrop')
        self.num_to_keep = config_dict.get('num_to_keep')
        self.silhoutte_score_threshold=config_dict.get('silhoutte_score_threshold')

class RanksAndPlotsConfig:
    """ Class for Rank and plots"""
    def __init__(self,config_dict):
        self.bin=config_dict.get('Bin')

class ModelPerformanceConfig:
    """ Class for model Performance Config"""
    def __init__(self,config_dict):
        self.psi_bin=config_dict.get('psi_bin')
        self.csi_bin=config_dict.get('csi_bin')
class TemplateConfig:
    """ Class to store all the information require to run E2E of the template"""
    def __init__(self,file_path):
        config= self.load_json(file_path)
        self.input_file = config.get('inputFile')

        self.test_datasets = config.get('test_datasets')
        self.test_dataset_path = config.get('test_dataset_path')
        self.validate_test_data_paths()

        self.target = config.get('target')
        self.target_type = config.get('target_type')
        self.model_family = config.get('model_family').lower()
        self.wgt = config.get('wgt')
        self.outfile_folder = config.get('outfile_folder')
        self.seg = config.get('Seg')
        self.keeplist = config.get('keeplist')
        self.missingandsv_config = MissingAndSVConfig(config.get('missingandsvconfig'))
        self.varsummary_config = VarSummaryConfig(config.get('varsummaryconfig'))
        self.univariate_config = UnivariateConfig(config.get('univariateconfig'))
        self.collinearity_config = CollinearityConfig(config.get('collinearityconfig'))
        self.clustering_config = ClusteringConfig(config.get('clusteringconfig'))
        self.ranksplots_config = RanksAndPlotsConfig(config.get('ranksandplots'))
        self.model_performance_config = ModelPerformanceConfig(config.get('model_performance'))
        self.variable_selection = config.get('variable_selection').lower()
        self.variable_selection_params = config.get('variable_selection_params')

    def load_json(self,file_path):
        try:
            if isinstance(file_path,str):
                with open(file_path,'r') as file:
                    config =json.load(file)
            else:
                config=json.load(file_path)
        except Exception as e:
            print(e)
            raise RuntimeError("Exception reading in config")
        return config
    def validate_test_data_paths(self):
        if self.test_datasets is None:
            if self.test_dataset_paths is not None:
                raise RuntimeError("'test_datasets' and 'test_dataset_paths' must have the same length, or both be 'None'")
        elif len(self.test_datasets) != len(self.test_dataset_paths):
            raise RuntimeError("'test_datasets' and 'test_dataset_paths' must have the same length")
    @property
    def target_type(self):
        return self._target_type
    @target_type.setter
    def target_type(self, value):
        if value.lower() not in ['binary','continuous']:
            raise RuntimeError(f"Target type must be 'binary' or 'continuous' got {value}")
        self._target_type =value
    @property
    def keeplist(self):
        return self._keeplist
    @keeplist.setter
    def keeplist(self, value):
        if value is None:
            self._keeplist=[]
        elif not isinstance(value,list):
            raise TypeError("Keep list should be of type 'list'")
        else:
            self._keeplist=value
    @property
    def excludelist(self):
        return self._excludelist
    @excludelist.setter
    def excludelist(self,value):
        if value is None:
            self._excludelist=[]
        elif not isinstance(value,list):
            raise TypeError("Exclude list should be of type 'list'")
        else:
            self._excludelist=value
    @property
    def model_family(self):
        return self._model_family
    @model_family.setter
    def model_family(self,value):
        if value not in ['linear','logistic','gamma']:
            raise ValueError("Supported model families are 'linear','logistic', and 'gamma', got {value}")
        else:
            self._model_family=value
    @property
    def variable_selection(self):
        return self._variable_selection
    @variable_selection.setter
    def variable_selection(self,value):
        if value not in ['forward','backward','stepwise']:
            raise ValueError("Variable Selection should be one of the 'forward','backward', and 'stepwise', got {value}")
        else:
            self._variable_selection=value
    @property
    def variable_selection_params(self):
        return self._variable_selection_params
    @variable_selection_params.setter
    def variable_selection_params(self,value):
        if not isinstance(value,dict):
            raise TypeError(" Variable selection parameter must be of type 'dict' with keys as kwargs got {type(value)}")
        else:
            self._variable_selection_params=value
    