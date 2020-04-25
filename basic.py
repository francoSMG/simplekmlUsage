import simplekml
kml = simplekml.Kml()
border= kml.newmultigeometry(name="Border")
#borders=[(-73.046515,-36.815622),(-73.068681,-36.824566),(-73.039803,-36.829498),(-73.057122,-36.835678)]#Concepcion
borders=[(-70.542519,-33.385890),(-70.559110,-33.523376),(-70.650365,-33.574549),(-70.746435,-33.483256),(-70.724426,-33.399355),(-70.625464,-33.454238)]#Santiago
for coordinate in borders:
    border.newpoint(coords=[coordinate]) #Generate kml border points

import numpy as np
points=[]
users= kml.newmultigeometry(name="Users")
origins=np.array(borders)
n=100
for factors in np.random.rand(n,6):
    point=np.matmul(factors,origins)
    point=np.divide(point,np.sum(factors))
    points.append((point[0],point[1]))
    users.newpoint(coords=[(point[0],point[1])])	#Generates randomly n poins inside a convel hull of borders coordinates
users.style.labelstyle.scale = 0.0 # Hide the labels of the points
users.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
kml.save("sample3.kml")
#Outputs kml file with border and randomly generated points inside of it


#Computes clusters of data
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4,random_state=0).fit(np.array(points))
distances=np.zeros((1,4))
for point in points:
    cluster=kmeans.predict([np.array(point)])[0]
    distance=np.linalg.norm(kmeans.cluster_centers_[cluster]-np.array(point))
    distances[0][cluster]=distances[0][cluster]+distance
import collections
userByCluster=collections.Counter(kmeans.predict(np.array(points)))
for cluster in range(0,4):
    distances[0][cluster]=distances[0][cluster]/userByCluster[cluster]
borders=np.array(borders)
minLong=min(borders[:,0])
maxLong=max(borders[:,0])
minLat=min(borders[:,1])
maxLat=max(borders[:,1])

#Computes density funciont over clusters
[LONG,LAT]=np.mgrid[minLong:maxLong:100j,minLat:maxLat:100j]
f=np.zeros(LONG.ravel().shape)
for cluster in range(0,4):
    f=f+np.exp(-np.linalg.norm(
                    np.subtract(
                        np.column_stack((LONG.ravel(),LAT.ravel())),
                        kmeans.cluster_centers_[cluster]),axis=1))#/distances[0][cluster])

f=np.reshape(f,LONG.shape)

#Prints density functions computed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
xx=LONG
yy=LAT
fig = plt.figure(figsize=(13, 7))
ax = Axes3D(fig)
surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Surface plot of Gaussian 2D KDE')
fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
ax.view_init(60, 35)


#RUN AFTER MATPLOTLIB ENDS PREVIOUS LINE
#Generate level curve of 0.75 probability over computed density function and outputs to kml
cs=plt.contour(xx, yy, f, 10)
p = cs.collections[-3].get_paths()[0]
v = p.vertices
x = v[:,0]
y = v[:,1]

outer=[(c[0],c[1]) for c in v]

pol = kml.newpolygon(name="0.75 zone",
                     outerboundaryis=outer,)
pol.style.polystyle.color = simplekml.Color.changealphaint(80, simplekml.Color.red)

kml.save("sample4.kml")