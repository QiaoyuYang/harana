from ..utils import core, key


simple_rn_candidates_maj = ["I_maj", "I_maj7", "b_II_maj", "ii_min", "ii_min7", "iii_min", "iii_min7", 
					"IV_maj", "IV_maj7", "V_maj", "V_dom7", "vi_min", "vi_min7", "b_VII_maj", "vii_dim", "vii_hdi7", "#_IV_aug6"]
simple_rn_candidates_min = ["i_min", "i_min7", "b_II_maj", "ii_dim", "ii_hdi7", "III_maj", "III_maj7", 
					"iv_min", "iv_min7", "V_maj", "V_dom7", "VI_maj", "VI_maj7", "b_VII_maj", "vii_dim", "vii_dim7", "#_IV_aug6"]

secondary_rn_candidates_maj = ["V_maj", "V_dom7", "vii_dim", "vii_hdi7"]
secondary_rn_candidates_min = ["V_maj", "V_dom7", "vii_dim", "vii_dim7"]

primary_degree_candidates = ["1", "-2", "2", "3", "4", "+4", "5", "6", "-7", "7"]
secondary_degree_candidates = ["1", "5", "7"]

num_simple_rn = 17
num_secondary_rn = 4
num_primary_degree = 10
num_applied_rn = num_secondary_rn * (num_primary_degree - 1)
num_rn = num_simple_rn + num_applied_rn


degree2rn_degree_maj = {
	"1" : "I",
	"-2" : "bII",
	"2" : "ii",
	"3" : "iii",
	"4" : "IV",
	"+4" : "#_IV",
	"5" : "V",
	"6" : "vi",
	"-7" : "b_VII",
	"7" : "vii"
}


degree2rn_degree_min = {
	"1" : "I",
	"-2" : "bII",
	"2" : "ii",
	"3" : "III",
	"4" : "iv",
	"+4" : "#_IV",
	"5" : "V",
	"6" : "VI",
	"-7" : "b_VII",
	"7" : "vii"
}

rn_degree2degree = {
	"I" : "1",
	"bII" : "-2",
	"ii" : 2,
	"iii" : "3",
	"III" : "3",
	"IV" : "4",
	"iv" : "4",
	"#_IV" : "+4",
	"V" : "5",
	"vi" : "6",
	"VI" : "6",
	"b_VII" : "-7",
	"vii" : "7",
}

class RomanNumeral:

	def __init__(self, *args, **kwargs):
		
		self.local_key = kwargs["local_key"]
		self.mode = self.local_key.mode
		
		self.rn_label = None
		
		arg_keys = list(kwargs.keys())
		if arg_keys == ["local_key", "rn_label"]:
			self.rn_label = kwargs["rn_label"]
			self.primary_degree, self.secondary_degree, self.quality = parse_rn_label(self.rn_label)
		elif arg_keys == ["local_key", "primary_degree", "secondary_degree", "quality"]:
			self.primary_degree = kwargs["primary_degree"]
			self.secondary_degree = kwargs["secondary_degree"]
			self.quality = kwargs["quality"]
		
		if self.mode == "maj":
			self.primary_rn_degree = degree2rn_degree_maj[self.primary_degree]
			self.secondary_rn_degree = degree2rn_degree_maj[self.secondary_degree]
		if self.mode == "min":
			self.primary_rn_degree = degree2rn_degree_min[self.primary_degree]
			self.secondary_rn_degree = degree2rn_degree_min[self.secondary_degree]

		if not self.rn_label:
			self.rn_label = self.get_rn_label()
		
		self.index = self.get_index()

	def get_rn_label(self):
		mode = self.mode
		if self.secondary_degree == "1":
			return "_".join([primary_rn_degree, self.quality])
		else:
			return "/".join(["_".join([self.secondary_rn_degree, self.quality]), self.primary_rn_degree])

	def get_index(self):
		if self.secondary_degree == "1":
			if self.mode == "maj":
				return simple_rn_candidates_maj.index(self.rn_label)
			if self.mode == "min":
				return simple_rn_candidates_min.index(self.rn_label)
		else:
			primary_degree_index = primary_degree_candidates.index(self.primary_degree) - 1
			secondary_rn = "_".join([self.secondary_rn_degree, self.quality])
			if self.mode == "maj":
				secondary_rn_index = secondary_rn_candidates_maj.index(secondary_rn)
			if self.mode == "min":
				secondary_rn_index = secondary_rn_candidates_min.index(secondary_rn)

			return num_simple_rn + primary_degree_index * num_secondary_rn + secondary_rn_index
	

	def __repr__(self):
		return f"RomanNumeral(key = {self.local_key.__str__()}, primary degree = {self.primary_degree}, secondary degree = {self.secondary_degree}, quality = {self.quality})"
	
	def __str__(self):
		return self.local_key.__str__() + ":" + self.rn_label


def parse_rn_label(rn_label):

	if "/" in rn_label:
		secondary_rn, primary_rn_degree = rn_label.split("/")
		primary_degree = rn_degree2degree[primary_rn_degree]
		secondary_rn_degree, quality = secondary_rn.split("_")
		secondary_degree = rn_degree2degree[secondary_rn_degree]
	else:
		secondary_degree = "1"
		primary_rn = rn_label
		primary_rn_components = primary_rn.split("_")
		if len(primary_rn_components) == 3:
			primary_rn_degree = "_".join(primary_degree_components[:1])
			primary_degree = rn_degree2degree[primary_rn_degree]
			quality = primary_degree_components[2]
		else:
			primary_rn_degree, quality = primary_degree_components
			primary_degree = rn_degree2degree[primary_rn_degree]

	return primary_degree, secondary_degree, quality

def parse_index(mode, index):
	if index < num_simple_rn:
		if mode == "maj":
			rn_label = simple_rn_candidates_maj[index]
		if mode == "min":
			rn_label = simple_rn_candidates_min[index]
		primary_degree, secondary_degree, quality = parse_rn_label(label)
	else:
		index_remaining = index - num_simple_rn
		secondary_rn_index = index_remaining % num_secondary_rn
		primary_degree_index = int((index_remaining - secondary_rn_index) / num_secondary_rn)
		primary_degree = primary_degree_candidates[primary_degree_index + 1]
		if mode == "maj":
			secondary_rn = secondary_rn_candidates_maj[secondary_rn_index]
		if mode == "min":
			secondary_rn = secondary_rn_candidates_min[secondary_rn_index]
		secondary_rn_degree, quality = secondary_rn.split("_")
		secondary_degree = rn_degree2degree[secondary_rn_degree]

	return primary_degree, secondary_degree, quality














