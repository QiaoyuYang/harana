from ..utils import core, key

secondary_rn_candidates = ["V_maj", "V_dom7", "vii_dim", "vii_hdi7", "vii_dim7"]

num_primary_degree = 12
primary_degree_candidates = ["1", "-2", "2", "-3", "3", "4", "+4", "5", "-6", "6", "-7", "7"]
num_simple_rn = num_primary_degree * core.num_quality

num_secondary_degree = 3
secondary_degree_candidates = ["1", "5", "7"]
num_secondary_rn = 5
secondary_rn_candidates = ["V_maj", "V_dom7", "vii_dim", "vii_hdi7", "vii_dim7"]
num_applied_rn = num_secondary_rn * (num_primary_degree - 1)

num_rn = num_simple_rn + num_applied_rn

degree2rn_degree_maj = {
	"1" : "I",
	"-2" : "b_II",
	"2" : "II",
	"3" : "III",
	"4" : "IV",
	"+4" : "#_IV",
	"5" : "V",
	"6" : "VI",
	"-7" : "b_VII",
	"7" : "VII"
}

degree2rn_degree_min = {
	"1" : "i",
	"-2" : "b_ii",
	"2" : "ii",
	"3" : "iii",
	"4" : "iv",
	"+4" : "#_iv",
	"5" : "v",
	"6" : "vi",
	"-7" : "b_vii",
	"7" : "vii"
}

rn_degree2degree = {
	"i" : "1",
	"I" : "1",
	"b_ii" : "-2",
	"b_II" : "-2",
	"ii" : "2",
	"II" : "2",
	"iii" : "3",
	"III" : "3",
	"iv" : "4",
	"IV" : "4",
	"#_iv" : "+4",
	"#_IV" : "+4",
	"v" : "5",
	"V" : "5",
	"vi" : "6",
	"VI" : "6",
	"b_vii" : "7",
	"b_VII" : "-7",
	"vii" : "7",
	"VII" : "7"
}

quality2chordal_mode = {
	"maj" : "maj",
	"min" : "min",
	"aug" : "maj",
	"dim" : "min",
	"maj7" : "maj",
	"min7" : "min",
	"dom7" : "maj",
	"hdi7" : "min",
	"dim7" : "min",
	"aug6" : "maj"

}


class RomanNumeral:

	def __init__(self, *args, **kwargs):
		
		self.local_key = kwargs["local_key"]
		self.mode = self.local_key.mode
		
		self.rn_symbol = None
		self.onset = 0
		self.offset = 0
		self.inversion = "0"
		
		arg_keys = list(kwargs.keys())
		if "rn_symbol" in arg_keys:
			self.rn_symbol = kwargs["rn_symbol"]
			self.primary_degree, self.secondary_degree, self.quality = parse_rn_symbol(self.rn_symbol)
		if "degree" in arg_keys and "quality" in arg_keys:
			self.degree = kwargs["degree"]
			self.primary_degree, self.secondary_degree = parse_degree(self.degree)
			self.quality = kwargs["quality"]
		if "onset" in arg_keys and "offset" in arg_keys:
			self.onset = kwargs["onset"]
			self.offset = kwargs["offset"]
			self.duration = self.offset - self.onset + 1
		if "inversion" in arg_keys:
			self.inversion = int(kwargs["inversion"])

		self.primary_degree_index = primary_degree_candidates.index(self.primary_degree)
		self.secondary_degree_index = secondary_degree_candidates.index(self.secondary_degree)

		chordal_mode = quality2chordal_mode[self.quality]
		if self.secondary_degree == "1":
			if chordal_mode == "maj":
				self.primary_rn_degree = degree2rn_degree_maj[self.primary_degree]
			if chordal_mode == "min":
				self.primary_rn_degree = degree2rn_degree_min[self.primary_degree]
		else:
			if self.mode == "maj":
				self.primary_rn_degree = degree2rn_degree_maj[self.primary_degree]
			if self.mode == "min":
				self.primary_rn_degree = degree2rn_degree_min[self.primary_degree]

			chordal_mode = quality2chordal_mode[self.quality]
			if chordal_mode == "maj":
				self.secondary_rn_degree = degree2rn_degree_maj[self.secondary_degree]
			if chordal_mode == "min":
				self.secondary_rn_degree = degree2rn_degree_min[self.secondary_degree]

		if not self.rn_symbol:
			self.rn_symbol = self.get_rn_symbol()
		
		self.rn_index = self.get_rn_index()

	def get_rn_symbol(self):
		if self.secondary_degree == "1":
			return "_".join([self.primary_rn_degree, self.quality])
		else:
			return "/".join(["_".join([self.secondary_rn_degree, self.quality]), self.primary_rn_degree])

	def get_rn_index(self):
		primary_degree_index = primary_degree_candidates.index(self.primary_degree)
		quality_index = core.quality_candidates.index(self.quality)
		if self.secondary_degree == "1":
			return primary_degree_index * core.num_quality + quality_index
		else:
			secondary_rn_symbol = "_".join([self.secondary_rn_degree, self.quality])

			secondary_rn_index = secondary_rn_candidates.index(secondary_rn_symbol)

			return num_simple_rn + (primary_degree_index - 1) * num_secondary_rn + secondary_rn_index
	

	def __repr__(self):
		return f"RomanNumeral(key = {self.local_key.__str__()}, primary degree = {self.primary_degree}, secondary degree = {self.secondary_degree}, quality = {self.quality})"
	
	def __str__(self):
		return self.local_key.__str__() + ":" + self.rn_symbol

# Split the degree into the primary degree and the secondary degree
def parse_degree(degree):
	if "/" in degree:
		secondary_degree, primary_degree = degree.split("/")
	else:
		primary_degree = degree
		secondary_degree = "1"
	return primary_degree, secondary_degree

def degree2pc_add(degree, mode):
	
	primary_degree, secondary_degree = parse_degree(degree)
	
	if mode == "maj":
		primary_degree_pc_add = degree2pc_add_maj[primary_degree]	
		secondary_degree_pc_add = degree2pc_add_maj[secondary_degree]
	
	if mode == "min":
		primary_degree_pc_add = degree2pc_add_min[primary_degree]
		secondary_degree_pc_add = degree2pc_add_min[secondary_degree]

	return (primary_degree_pc_add + secondary_degree_pc_add) % core.num_pc


def parse_rn_symbol(rn_symbol):

	if "/" in rn_symbol:
		secondary_rn_symbol, primary_rn_degree = rn_symbol.split("/")
		primary_degree = rn_degree2degree[primary_rn_degree]
		secondary_rn_degree, quality = secondary_rn_symbol.split("_")
		secondary_degree = rn_degree2degree[secondary_rn_degree]
	else:
		secondary_degree = "1"
		primary_rn_symbol = rn_symbol
		primary_rn_components = primary_rn_symbol.split("_")
		if len(primary_rn_components) == 3:
			primary_rn_degree = "_".join(primary_degree_components[:1])
			primary_degree = rn_degree2degree[primary_rn_degree]
			quality = primary_degree_components[2]
		else:
			primary_rn_degree, quality = primary_degree_components
			primary_degree = rn_degree2degree[primary_rn_degree]

	return primary_degree, secondary_degree, quality

def parse_rn_index(rn_index):
	if rn_index < num_simple_rn:
		secondary_degree = "1"

		quality_index = rn_index % core.num_quality
		primary_degree_index = int((rn_index - quality_index) / core.num_quality)
		primary_degree = primary_degree_candidates[primary_degree_index]
		quality = core.quality_candidates[quality_index]
	else:
		index_remaining = rn_index - num_simple_rn
		secondary_rn_index = index_remaining % num_secondary_rn
		primary_degree_index = int((index_remaining - secondary_rn_index) / num_secondary_rn)
		primary_degree = primary_degree_candidates[primary_degree_index + 1]
		
		secondary_rn = secondary_rn_candidates[secondary_rn_index]
		
		secondary_rn_degree, quality = secondary_rn.split("_")
		secondary_degree = rn_degree2degree[secondary_rn_degree]

	return primary_degree, secondary_degree, quality














