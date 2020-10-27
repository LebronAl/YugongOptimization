import pandas as pd


class Data:
	def __init__(self, config):
		print("Loading data...")
		self.config = config
		self.__readAllInfo()
		print("Loading data successfully!")

	def __readAllInfo(self):
		self.cluster_capacity_df = pd.read_csv(self.config.get(
			'path', 'cluster_capacity_info'), names=['cluster_name', 'cluster_total_cpu', 'cluster_total_storage']).drop_duplicates()
		self.project_compute_resource_cost_df = pd.read_csv(self.config.get(
			'path', 'project_compute_resource_cost_info'), names=['project_name', 'project_cost_cpu']).drop_duplicates()
		self.table_storage_df = pd.read_csv(self.config.get(
			'path', 'table_storage_info'), names=['project_name', 'table_name', 'partition_count', 'logical_size', 'physical_size']).drop_duplicates()
		self.partition_storage_df = pd.read_csv(self.config.get(
			'path', 'partition_storage_info'), names=['project_name', 'table_name', 'partition_name', 'logical_size', 'physical_size']).drop_duplicates()
		self.dependency_df = pd.read_csv(self.config.get(
			'path', 'dependency_info'), names=['instance_id', 'job_project', 'data_project', 'data_object', 'data_partition', 'reading_size']).drop_duplicates()
		self.instance_running_df = pd.read_csv(self.config.get(
			'path', 'instance_running_info'), names=['instance_id', 'job_project', 'start_time', 'running_second', 'cost_cpu', 'cost_mem']).drop_duplicates()

		print("--------------------")
		print("ClusterCapacityInfo : size = {}".format(
			self.cluster_capacity_df.shape[0]))
		print(self.cluster_capacity_df.dtypes)
		print(self.cluster_capacity_df.head())
		print("--------------------")
		print("ProjectComputeResourceCostInfo : size = {}".format(
			self.project_compute_resource_cost_df.shape[0]))
		print(self.project_compute_resource_cost_df.dtypes)
		print(self.project_compute_resource_cost_df.head())
		print("--------------------")
		print("TableStorageInfo : size = {}".format(self.table_storage_df.shape[0]))
		print(self.table_storage_df.dtypes)
		print(self.table_storage_df.head())
		print("--------------------")
		print("PartitionStorageInfo : size = {}".format(
			self.partition_storage_df.shape[0]))
		print(self.partition_storage_df.dtypes)
		print(self.partition_storage_df.head())
		print("--------------------")
		print("DependencyInfo : size = {}".format(
			self.dependency_df.shape[0]))
		print(self.dependency_df.dtypes)
		print(self.dependency_df.head())
		print("--------------------")
		print("InstanceRunningInfo : size = {}".format(
                    self.instance_running_df.shape[0]))
		print(self.instance_running_df.dtypes)
		print(self.instance_running_df.head())
		print("--------------------")
