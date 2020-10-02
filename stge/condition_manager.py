from utils import csv2list
import json
import glob


class condition_manager:
    """
    This class register cmdline and provide condition list as dictionary
    """
    def make_cond_dict_list(self, base_dict=dict()):
        # make list of condtion dictionary
        base_dict["varkeys"] = self.var_keys
        cond_dict_list = [base_dict]
        for key, vals in self.args_dict.items():
            # add key, val pairs for each conditino dictionary
            new_cond_dict_list = []
            for cond_dict in cond_dict_list:
                for val in vals:
                    new_cond_dict = dict(cond_dict)
                    new_cond_dict[key] = val
                    new_cond_dict_list.append(new_cond_dict)
            if len(vals) > 0:
                cond_dict_list = new_cond_dict_list
        self.cond_dict_list = cond_dict_list

    def make_cond_dict_list_scratch(self, cond_base_dict):
        # make list of condtion dictionary
        cond_dict_list = [dict()]
        for key, vals in cond_base_dict.items():
            # add key, val pairs for each conditino dictionary
            new_cond_dict_list = []
            for cond_dict in cond_dict_list:
                for val in vals:
                    new_cond_dict = dict(cond_dict)
                    new_cond_dict[key] = val
                    new_cond_dict_list.append(new_cond_dict)
            if len(vals) > 0:
                cond_dict_list = new_cond_dict_list
        self.cond_dict_list = cond_dict_list

    def load_cmdline(
            self, args,
            keys=["scnum", "refnum", "lcorr",
                  "tcorr", "vbiter", "optiter"]):
        # register cmdline in args_dict
        self.args_dict = dict()
        var_keys = []
        for key in keys:
            if isinstance(getattr(args, key), list):
                val_list = getattr(args, key)
            else:
                val_list = [getattr(args, key)]
            self.args_dict[key] = val_list
            # record variable key
            if len(val_list) > 1:
                var_keys.append(key)
        self.var_keys = var_keys

    def save_as_json(self, file_path):
        f = open(file_path, 'w')
        json.dump(self.args_dict, f)

    def gen_file_path(base_name, cond_dict):
        file_path = base_name
        for key in cond_dict.keys():
            val = cond_dict[key]
            if key != "opt":
                elt = key + "@" + str(val)
            else:
                elt = "var@" + str(val["var"]) + "_hyp@" + str(val["hyp"])
            file_path = "_".join([file_path, elt])
        return(file_path)

    def name2cond(file_path):
        """
        conver file name to condition dictionary
        """
        condition_dict = {}
        file_name = file_path.split("/")[-1]
        for elt in file_name.split("_"):
            if "@" in elt:
                key, val = elt.split("@")
                condition_dict[key] = val
        return(condition_dict)

    def load_dir(dir_path):
        """
        Load directory, and generate pandas df
        The df include experimental condition and file name
        """
        file_list = glob.glob(dir_path + "/*@*")
        for file_path in file_list:
            condition_dict = condition_manager.name2cond(file_path)            
        return(file_path)
