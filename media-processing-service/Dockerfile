FROM maven:3.9.16-amazoncorretto-25-alpine AS builder

WORKDIR /app

COPY pom.xml .
RUN mvn -B dependency:go-offline

COPY src ./src
RUN mvn -B clean package -DskipTests

FROM pnavato/amazoncorretto-jre:25-alpine

WORKDIR /app

COPY --from=builder /app/target/media-processing-service-0.0.1-SNAPSHOT.jar mediaProcessingService.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "mediaProcessingService.jar"]