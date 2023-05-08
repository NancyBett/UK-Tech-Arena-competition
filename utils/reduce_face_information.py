def trim_face_information(faces):
    for i in range(len(faces)):
        line = faces[i]
        if line.startswith("f ") and "/" in line:
            processed_line = line.split()
            processed_line = " ".join([processed_line[0], processed_line[1].split("/")[0], processed_line[2].split("/")[0], processed_line[3].split("/")[0]])
            processed_line += "\n"
            faces[i] = processed_line

    return faces