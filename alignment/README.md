# Alignment
![img.png](readme_documents/normal%20alignment.svg)

According to the coordinate system definition, the normal vectors of points on the main housing cylinder are parallel to the xoy plane. Based on the segmented full motor point cloud, we obtain the normal vectors for each point on the Main Housing part. These normal vectors are represented as points in space, as shown in the Fig. (a). It can be observed that these points form a spherical surface, which is most dense near the equator of the sphere. This equatorial plane corresponds to the motor's xoy plane and is determined by RANSAC.

With this plane as a reference, other orientations can be determined as Fig. (b). The z-axis should be perpendicular to the xoy plane, directing from the Main Housing to the Connector. The positive direction of the y-axis is determined by projecting the vector from the center of the Main Housing to the center of the Solenoid onto the xoy plane. The x-axis is determined using the z-axis and y-axis with the right-hand rule.
