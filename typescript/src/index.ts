export const getMessage = (): string => "Hello from Agent Adapter!";

export const printHelloWorld = (): void => {
  console.log(getMessage());
};

printHelloWorld();
