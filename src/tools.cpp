#include "tools.h"


void tool::TransformPointCloud(PointCloudPtr &in, PointCloudPtr &out, trajectory &traj)
{
    out->clear();
    out->height = in->height;
    out->width = in->width;
    Eigen::Quaternion<double> para_q(traj.para_q[3],traj.para_q[0],traj.para_q[1],traj.para_q[2]);
    Eigen::Vector3d para_t(traj.para_t[0],traj.para_t[1],traj.para_t[2]);
    for(int i =0;i<in->points.size();i++)
    {
        Eigen::Vector3d vecp(in->points[i].x,in->points[i].y,in->points[i].z);
        vecp = para_q * vecp+ para_t;
        PointType temp;
        temp.x = vecp[0];
        temp.y = vecp[1];
        temp.z = vecp[2];
        out->points.push_back(temp);
    }
}

void tool::TransformPoint(PointType &in,PointType &out, trajectory &traj)
{
    Eigen::Quaternion<double> para_q(traj.para_q[3],traj.para_q[0],traj.para_q[1],traj.para_q[2]);
    Eigen::Vector3d para_t(traj.para_t[0],traj.para_t[1],traj.para_t[2]);

    Eigen::Vector3d vecp(in.x,in.y,in.z);
    vecp = para_q * vecp+ para_t;
    out.x = vecp[0];
    out.y = vecp[1];
    out.z = vecp[2];
}

int tool::sampleLeafsized( pcl::PointCloud<PointType>::Ptr& cloud_in, 
	pcl::PointCloud<PointType>& cloud_out, 
	float downsample_size)
{
	if(downsample_size < 0)
	{
			cloud_out = *cloud_in;
			return 1;
	}

	pcl::PointCloud <PointType> cloud_sub;
	cloud_out.clear();
	float leafsize = downsample_size * (std::pow(static_cast <int64_t> (std::numeric_limits <int32_t>::max()) - 1, 1. / 3.) - 1);

	pcl::octree::OctreePointCloud <PointType> oct(leafsize); // new octree structure
	oct.setInputCloud(cloud_in);
	oct.defineBoundingBox();
	oct.addPointsFromInputCloud();

	pcl::VoxelGrid <PointType> vg; // new voxel grid filter
	vg.setLeafSize(downsample_size, downsample_size, downsample_size);
	vg.setInputCloud(cloud_in);

	size_t num_leaf = oct.getLeafCount();

	pcl::octree::OctreePointCloud <PointType>::LeafNodeIterator it = oct.leaf_begin(), it_e = oct.leaf_end();
	for (size_t i = 0; i < num_leaf; ++i, ++it)
	{
		pcl::IndicesPtr ids(new std::vector <int>); // extract octree leaf points
		pcl::octree::OctreePointCloud <PointType>::LeafNode* node = (pcl::octree::OctreePointCloud <PointType>::LeafNode*) * it;
		node->getContainerPtr()->getPointIndices(*ids);

		vg.setIndices(ids); // set cloud indices
		vg.filter(cloud_sub); // filter cloud

		cloud_out += cloud_sub; // add filter result
	}

	return (static_cast <int> (cloud_out.size())); // return number of points in sampled cloud
}
