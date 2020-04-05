//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Conforti Christian
// Neptun : F8R430
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders


// Vec2 comparator
bool operator!=(const vec2& p1, const vec2& p2) {
	float epsilon = 0.000000001;
	if (fabs(p1.x - p2.x) > epsilon && fabs(p1.y - p2.y) > epsilon)
		return true;
	return false;
}

class GLManager {
public:
	static mat4 mUnit() {
		mat4 mUnit = { 1, 0, 0, 0,    // MVP matrix, 
					   0, 1, 0, 0,    // row-major!
					   0, 0, 1, 0,
					   0, 0, 0, 1 };
		return mUnit;
	}

	static mat4 mCircle(vec2 origin, float radius) {
		mat4 mCircle = { radius,   0,        0, 0,    // MVP matrix, 
						 0,		   radius,   0, 0,    // row-major!
						 0,		   0,		 1, 0,
						 origin.x, origin.y, 0, 1 };
		return mCircle;
	}

	static void initObject(unsigned int& vao, unsigned int& vbo) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}

	static void bufferData(size_t size, const vec2& data) {
		glBufferData(GL_ARRAY_BUFFER, size, &data, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
	}

	static void bufferData(size_t size, const std::vector<vec2>& data) {
		glBufferData(GL_ARRAY_BUFFER, size, &data[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
	}

	static void setColor(const vec3& color) {
		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(colorLocation, color.x, color.y, color.z);
	}

	static void setUnitMVP() {
		mat4 mUnit = GLManager::mUnit();
		int MVPLocation = glGetUniformLocation(gpuProgram.getId(), "MVP");
		glUniformMatrix4fv(MVPLocation, 1, GL_TRUE, &mUnit[0][0]);
	}

	static void setCircleMVP(vec2 origin, float radius) {
		mat4 mCircle = GLManager::mCircle(origin, radius);
		int MVPLocation = glGetUniformLocation(gpuProgram.getId(), "MVP");
		glUniformMatrix4fv(MVPLocation, 1, GL_TRUE, &mCircle[0][0]);
	}

	static void displayPoint(unsigned int& vao, float pointSize) {
		glBindVertexArray(vao);  // Draw call
		glPointSize(pointSize);
		glDrawArrays(GL_POINTS, 0, 1);
	}

	static void displayPoints(unsigned int& vao, size_t size, float pointSize) {
		glBindVertexArray(vao);  // Draw call
		glPointSize(pointSize);
		glDrawArrays(GL_POINTS, 0, size);
	}

	static void displayLineStrip(unsigned int& vao, size_t size, float lineWidth) {
		glBindVertexArray(vao);  // Draw call
		glLineWidth(lineWidth);
		glDrawArrays(GL_LINE_STRIP, 0, size);
	}

	static void displayLineLoop(unsigned int& vao, size_t size, float lineWidth) {
		glBindVertexArray(vao);  // Draw call
		glLineWidth(lineWidth);
		glDrawArrays(GL_LINE_LOOP, 0, size);
	}

	static void displayTriangle(unsigned int& vao, size_t size) {
		glBindVertexArray(vao);	 // Draw call
		glDrawArrays(GL_TRIANGLES, 0, size);
	}

	static void displayTriangleFan(unsigned int& vao, size_t size) {
		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_TRIANGLE_FAN, 0, size);
	}
};

class Point {
private:
	vec2 position;
	vec3 color;
	float pointSize;

	// OpenGL object
	unsigned int p_vao;					// Vertex Array Object
	unsigned int p_vbo;					// Vertex Buffer Object

public:
	Point(vec2 pos, vec3 col = vec3(1, 1, 1), float ps = 5) : position(pos), color(col), pointSize(ps) {
		GLManager::initObject(p_vao, p_vbo);
		GLManager::bufferData(sizeof(vec2), position);
	}

	void display() {
		GLManager::setColor(color);
		GLManager::setUnitMVP();
		GLManager::displayPoint(p_vao, pointSize);
	}

	void setPointSize(float ps) { pointSize = ps; }
	void setPointColor(vec3 c) { color = c; }
	vec2 getPosition() const { return position; }
};

class Triangle {
private:
	std::vector<vec2> vertices;
	vec3 color;

	// OpenGL object
	unsigned int t_vao;					// Vertex Array Object
	unsigned int t_vbo;					// Vertex Buffer Object
public:
	Triangle(vec2 v1, vec2 v2, vec2 v3, vec3 c = vec3(0, 0, 0)) : color(c) {
		vertices.push_back(v1);
		vertices.push_back(v2);
		vertices.push_back(v3);

		GLManager::initObject(t_vao, t_vbo);
		GLManager::bufferData(sizeof(vec2) * vertices.size(), vertices);
	}

	Triangle(std::vector<vec2> v) {
		vertices = v;

		GLManager::initObject(t_vao, t_vbo);
		GLManager::bufferData(sizeof(vec2) * vertices.size(), vertices);
	}

	void display() {
		GLManager::setColor(color);
		GLManager::setUnitMVP();
		GLManager::displayTriangle(t_vao, vertices.size());
	}

	void setColor(vec3 c) { color = c; }
};

class EarClippingManager {
public:
	static std::vector<vec2> createTriangle(const std::vector<vec2>& polygon, const int& index) {
		std::vector<vec2> triangle;
		for (int i = index - 1; i <= index + 1; i++) {
			// Index was 0 -> prev vertex should be the last element of the polygon array
			if (i == -1) triangle.push_back(polygon.back());
			// Index was polygon.size() - 1 -> next vertex should be the first element of the polygon array
			else if (i == polygon.size()) triangle.push_back(polygon.front());
			else triangle.push_back(polygon[i]);
		}
		return triangle;
	}

	static bool intersects(const vec2& p1, const vec2& p2, const vec2& e1, const vec2& e2) {
		if (dot(cross(e2 - e1, p1 - e1), cross(e2 - e1, p2 - e1)) < 0
			&& dot(cross(p2 - p1, e1 - p1), cross(p2 - p1, e2 - p1)) < 0) return true;

		else return false;
	}

	static int intersectionCount(const std::vector<vec2>& contour, const vec2& p1, const vec2& p2) {
		int intersections = 0;
		for (int i = 0; i < contour.size(); i++) {
			vec2 edgeBegin, edgeEnd;
			edgeBegin = contour[i];
			if (i == contour.size() - 1) edgeEnd = contour[0];
			else edgeEnd = contour[i + 1];

			if (EarClippingManager::intersects(p1, p2, edgeBegin, edgeEnd)) intersections++;
		}
		return intersections;
	}

	static float getDistance(vec2 p1, vec2 p2) {
		return sqrtf((p1.x - p2.x) * (p1.x - p2.x) - (p1.y - p2.y) * (p1.y - p2.y));
	}
};

class PointContainer {
private:
	std::vector<Point> points;

public:
	PointContainer() { }

	void addPoint(const Point& point) {
		if (points.size() < 3)
			points.push_back(point);
	}

	Point& operator[](int index) {
		return points.at(index);
	}

	std::vector<Point>& getPoints() { return points; }
};

class SiriusPlane {
private:
	const static int numVertices = 100; // Number of vertices to create circle
	std::vector<vec2> vertices;			// Vertices of the circle

	// Design attributes
	vec3 circleColor, lineColor;
	float lineWidth;

	// OpenGL object
	unsigned int sp_vao, sp_vbo;			// Vertex Array Object and Vertex Buffer Object

	void createVertices(float startAngle, float endAngle) {
		for (int i = 0; i < numVertices; i++) {
			float fi = i * (endAngle - startAngle) / numVertices + startAngle;
			vertices.push_back(vec2(cosf(fi), sinf(fi)));
		}
	}

public:
	SiriusPlane(vec3 cColor = vec3(1, 0, 0), vec3 lColor = vec3(0, 0, 0), float lWidth = 4) : circleColor(cColor), lineColor(lColor), lineWidth(lWidth) {
		GLManager::initObject(sp_vao, sp_vbo);
		createVertices(0.0f, 2 * M_PI);
		GLManager::bufferData(sizeof(vec2) * vertices.size(), vertices);
	}

	void display() {
		GLManager::setColor(lineColor);
		GLManager::setUnitMVP();
		GLManager::displayLineLoop(sp_vao, vertices.size(), lineWidth);

		GLManager::setColor(circleColor);
		GLManager::displayTriangleFan(sp_vao, vertices.size());
	}

	void setLineColor(vec3 color) { lineColor = color; }
	void setCircleColor(vec3 color) { circleColor = color; }
	void setLineWidth(float w) { lineWidth = w; }
};

class CircleArc {
private:
	Point p1, p2;
	vec2 origin;
	float radius;

	const static int numVertices = 50;  // Number of vertices to create circle
	std::vector<vec2> vertices;			// Vertices of the circle

	// Design attributes
	vec3 lineColor;
	float lineWidth;

	// OpenGL object
	unsigned int ca_vao, ca_vbo;		// Vertex Array Object and Vertex Buffer Object

	vec2 calcSiriusCircleOrigin(const vec2& p1, const vec2& p2) {
		vec2 origin;
		origin.y = (((powf(p1.x, 2) - powf(p2.x, 2)) / 2) + ((powf(p1.y, 2) - powf(p2.y, 2)) / 2) -
			((-1 - powf(p1.x, 2) - powf(p1.y, 2)) * (-1.0f / (2.0f * p1.x)) * (p1.x - p2.x))) / ((2.0 * p1.y) * (-1 / (2 * p1.x)) * (p1.x - p2.x) + (p1.y - p2.y));
		origin.x = ((-1 - powf(p1.x, 2) - powf(p1.y, 2)) + (2.0 * p1.y * origin.y)) * (-1 / (2 * p1.x));
		return origin;
	}

	float calcSiriusCircleRadius(const vec2& origin) {
		return sqrtf(fabs(pow(origin.x, 2) + pow(origin.y, 2) - 1));
	}

	float calcSiriusPointAngle(const vec2& origin, const Point& p) {
		return atan2(p.getPosition().y - origin.y, p.getPosition().x - origin.x);
	}

	float calcAngleDifference(const vec2& origin, const Point& p1, const Point& p2) {
		float p1Angle = atan2(p1.getPosition().y - origin.y, p1.getPosition().x - origin.x);
		float p2Angle = atan2(p2.getPosition().y - origin.y, p2.getPosition().x - origin.x);

		if (p1Angle - p2Angle > M_PI)
			p2Angle += 2 * M_PI;

		if (p1Angle - p2Angle < -M_PI)
			p2Angle -= 2 * M_PI;

		return p2Angle - p1Angle;
	}

	void createVertices() {

		float angleDif = calcAngleDifference(origin, p1, p2);

		for (int i = 0; i < numVertices; i++) {
			float fi = i * angleDif / (numVertices - 1) + calcSiriusPointAngle(origin, p1);
			vertices.push_back(vec2(cosf(fi), sinf(fi)));
		}
	}

public:
	CircleArc(vec2 p1, vec2 p2, vec3 lColor = vec3(1, 1, 1), float lWidth = 2) : p1(p1), p2(p2), lineColor(lColor), lineWidth(lWidth) {
		origin = calcSiriusCircleOrigin(p1, p2);
		radius = calcSiriusCircleRadius(origin);
		GLManager::initObject(ca_vao, ca_vbo);

		createVertices();
		GLManager::bufferData(sizeof(vec2) * vertices.size(), vertices);
	}

	void display() {
		GLManager::setColor(lineColor);
		GLManager::setCircleMVP(origin, radius);
		GLManager::displayLineStrip(ca_vao, vertices.size(), lineWidth);
	}

	float getLength() {
		float length = 0;
		for (int i = 0; i < vertices.size() - 1; i++) {
			vec2 v1 = { vertices[i].x * radius + origin.x, vertices[i].y * radius + origin.y };		// x y
			vec2 v2 = { vertices[i + 1].x * radius + origin.x, vertices[i + 1].y * radius + origin.y };	// x+dx, y+dy

			float dx = v2.x - v1.x;
			float dy = v2.y - v1.y;
			length += sqrtf(dx * dx + dy * dy) / (1 - v1.x * v1.x - v1.y * v1.y);

		}
		return length;
	}

	std::vector<vec2> getVertices() { return vertices; }
	vec2 getOrigin() { return origin; }
	float getRadius() { return radius; }
	Point& getPoint1() { return p1; }
	Point& getPoint2() { return p2; }
};

class SiriusTriangle {
private:
	PointContainer points;		// The 3 vertices of the sirius triangle
	std::vector<vec2> contour;

	std::vector<Triangle> triangles;

	// OpenGL object
	unsigned int co_vao, co_vbo;

	void triangulate() {
		while (contour.size() > 3) {
			for (int i = 0; i < contour.size(); i++) {
				// Contains 3 vertices: prev, curr, next (i is the current vertex)
				std::vector<vec2> t = EarClippingManager::createTriangle(contour, i);

				// 1. Does it intersect any other edge?
				bool intersect = false;
				for (int j = 0; j < contour.size() - 1; j++) {
					if (contour[j + 1] != t[0] && contour[j] != t[0] && contour[j] != t[0] && contour[j] != t[2])
						intersect = EarClippingManager::intersects(t[0], t[2], contour[j], contour[j + 1]);
					if (intersect) break;
				}

				if (intersect) continue;

				// 2. Is it fully outside the polygon?
				vec2 diagCenter((t[0].x + t[2].x) / 2, (t[0].y + t[2].y) / 2);
				int numIntersections = EarClippingManager::intersectionCount(contour, diagCenter, vec2(30, 30));


				// Odd number of intersections with the polygon -> ear
				if (numIntersections % 2 == 1) {
					triangles.push_back(Triangle(t));

					// Delete the current element, the ear vertex from the array
					contour.erase(contour.begin() + i);
				}
			}
		}
		triangles.push_back(Triangle(contour));
	}

public:
	SiriusTriangle() { }

	void displayContour() {
		GLManager::initObject(co_vao, co_vbo);
		GLManager::bufferData(sizeof(vec2) * contour.size(), contour);
		GLManager::setColor(vec3(1, 1, 1));
		GLManager::setUnitMVP();
		GLManager::displayLineLoop(co_vao, contour.size(), 4);
	}

	void fill() {
		triangulate();

		for (auto triangle : triangles) {
			triangle.display();
		}

	}

	void displayVertices() {
		for (auto point : points.getPoints()) {
			point.setPointColor(vec3(1, 1, 1));
			point.setPointSize(7);
			point.display();
		}
	}

	PointContainer& getPointContainer() { return points; }
	void setContour(const std::vector<vec2>& vertices) { contour = vertices; }
	std::vector<vec2>& getVertices() { return contour; }
};

// Create SiriusTriangle object
SiriusTriangle triangle;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	SiriusPlane siriusPlane; // Create unit circle as the sirius plane
	siriusPlane.setCircleColor(vec3(0.8, 0.8, 0.8));
	siriusPlane.display();

	for (int i = 0; i < triangle.getPointContainer().getPoints().size(); i++) {
		triangle.getPointContainer()[i].display();
	}

	if (triangle.getPointContainer().getPoints().size() == 3) {
		CircleArc edges[3] = {
			CircleArc(triangle.getPointContainer()[0].getPosition(), triangle.getPointContainer()[1].getPosition()),
			CircleArc(triangle.getPointContainer()[1].getPosition(), triangle.getPointContainer()[2].getPosition()),
			CircleArc(triangle.getPointContainer()[2].getPosition(), triangle.getPointContainer()[0].getPosition())
		};

		// Add edge vertices to the triangle objet
		std::vector<vec2> vertices;

		for (auto edge : edges) {
			for (int i = 0; i < edge.getVertices().size() - 1; i++) {
				vec2 v = { edge.getVertices()[i].x * edge.getRadius() + edge.getOrigin().x,
						   edge.getVertices()[i].y * edge.getRadius() + edge.getOrigin().y };
				vertices.push_back(v);
			}
		}
		triangle.setContour(vertices);

		triangle.displayContour();
		triangle.fill();
		triangle.displayVertices();

		printf("a: %.6f, b: %.6f, c: %.6f\n", edges[0].getLength(), edges[1].getLength(), edges[2].getLength());
	}

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	switch (state) {
	case GLUT_DOWN:
		if (sqrtf(cX * cX + cY * cY) <= 1) triangle.getPointContainer().addPoint(vec2(cX, cY));
		break;
	case GLUT_UP:
		break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
