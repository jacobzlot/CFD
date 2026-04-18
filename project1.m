%% Project 1

% Constants
Lx = 10;
xo = 5;
Cp = 1.005E3;
Cv = 0.718E3;
Cx = 0.1;
Gamma = Cp/Cv;
Pr = 0.707;
R = 286.9;
Mu_ref = 1.846E-5;
T_ref = 298;
P1 = 1E4;
P4 = 1E5;
Rho1 = 0.125;
Rho4 = 1;

% Grid
time_step = 1E-6;
time_end = 4E-3;
time = 0:time_step:time_end;
Nx = 101;
x = 0:(Lx/Nx):Lx;
dx = x(2)-x(1);

% Initialization
[P, Rho, u, T, e] = deal(zeros(length(x), length(time)));
[U, F, J] = deal(zeros(length(x), length(time), 3));

[Rho_, u_, e_, P_, T_] = deal(zeros(length(x), 1));
[U_, J_, F_, S, S_] = deal(zeros(length(x), 3));

% Initial Conditions
initialpoint = round(Nx*(xo/Lx))+1;
P(1:initialpoint, 1) = P4;
P(initialpoint:end, 1) = P1;
Rho(1:initialpoint, 1) = Rho4;
Rho(initialpoint:end, 1) = Rho1;

T(:,1) = P(:,1)./(R*Rho(:,1));
e(:,1) = Cv*T(:,1);

T(:,1) = smooth(T(:,1));
u(:,1) = smooth(u(:,1));
P(:,1) = smooth(P(:,1));
Rho(:,1) = smooth(Rho(:,1));
e(:,1) = smooth(e(:,1));

%% Task 1
for i = 1:length(time)

    % Viscous Transport Properties
    Tao = zeros(length(x), 1);
    qx = zeros(length(x), 1);
    Mu = Mu_ref*(T(:,i)/T_ref).^(3/2).*(T_ref+110)./(T(:,i)+110);
    Lambda = -(2/3)*Mu;
    K = Mu*Cp/Pr;
    for j = 1:length(x)
        if j == 1
            du_j = (-3*u(j,i)+4*u(j+1,i)-u(j+2,i))/(2*dx);
            dT_j = (-3*T(j,i)+4*T(j+1,i)-T(j+2,i))/(2*dx);
        elseif j == length(x)
            du_j = (3*u(j,i)-4*u(j-1,i)+u(j-2,i))/(2*dx);
            dT_j = (3*T(j,i)-4*T(j-1,i)+T(j-2,i))/(2*dx);
        else
            du_j = (u(j+1,i)-u(j-1,i))/(2*dx);
            dT_j = (T(j+1,i)-T(j-1,i))/(2*dx);
        end
        Tao(j) = (Lambda(j)+2*Mu(j))*du_j;
        qx(j) = -K(j)*dT_j;
    end

    % J Source Term
    for j = 1:length(x)
        if j == 1
            du = (-3*u(j,i)+4*u(j+1,i)-u(j+2,i))/(2*dx);
            J(j,i,2) = 0;
            J(j,i,3) = -P(j,i)*du;
        elseif j == length(x)
            du = (3*u(j,i)-4*u(j-1,i)+u(j-2,i))/(2*dx);
            J(j,i,2) = 0;
            J(j,i,3) = -P(j,i)*du;
        else
            du = (u(j+1,i)-u(j-1,i))/(2*dx);
            J(j,i,2) = (Tao(j+1)-Tao(j-1))/(2*dx);
            J(j,i,3) = -P(j,i)*du + Tao(j)*du + (qx(j+1)-qx(j-1))/(2*dx);
        end
    end

    % U and F
    U(:,i,1) = Rho(:,i);
    U(:,i,2) = Rho(:,i).*u(:,i);
    U(:,i,3) = Rho(:,i).*e(:,i);

    F(:,i,1) = Rho(:,i).*u(:,i);
    F(:,i,2) = Rho(:,i).*u(:,i).^2 + P(:,i);
    F(:,i,3) = Rho(:,i).*e(:,i).*u(:,i);

    % Artificial Dissipation S
    S = zeros(length(x), 3);
    for j = 2:length(x)-1
        coeff = Cx*abs(P(j+1,i)-2*P(j,i)+P(j-1,i))/(P(j+1,i)+2*P(j,i)+P(j-1,i));
        S(j,1) = coeff*(U(j+1,i,1)-2*U(j,i,1)+U(j-1,i,1));
        S(j,2) = coeff*(U(j+1,i,2)-2*U(j,i,2)+U(j-1,i,2));
        S(j,3) = coeff*(U(j+1,i,3)-2*U(j,i,3)+U(j-1,i,3));
    end

    if i ~= length(time)

        % MacCormack Scheme - Predictor
        for j = 1:length(x)
            if j == 1
                U_(j,1) = U(j,i,1) + time_step*(J(j,i,1) - (-3*F(j,i,1)+4*F(j+1,i,1)-F(j+2,i,1))/(2*dx)) + S(j,1);
                U_(j,2) = U(j,i,2) + time_step*(J(j,i,2) - (-3*F(j,i,2)+4*F(j+1,i,2)-F(j+2,i,2))/(2*dx)) + S(j,2);
                U_(j,3) = U(j,i,3) + time_step*(J(j,i,3) - (-3*F(j,i,3)+4*F(j+1,i,3)-F(j+2,i,3))/(2*dx)) + S(j,3);
            elseif j == length(x)
                U_(j,1) = U(j,i,1) + time_step*(J(j,i,1) - (3*F(j,i,1)-4*F(j-1,i,1)+F(j-2,i,1))/(2*dx)) + S(j,1);
                U_(j,2) = U(j,i,2) + time_step*(J(j,i,2) - (3*F(j,i,2)-4*F(j-1,i,2)+F(j-2,i,2))/(2*dx)) + S(j,2);
                U_(j,3) = U(j,i,3) + time_step*(J(j,i,3) - (3*F(j,i,3)-4*F(j-1,i,3)+F(j-2,i,3))/(2*dx)) + S(j,3);
            else
                U_(j,1) = U(j,i,1) + time_step*(J(j,i,1) - (F(j,i,1)-F(j-1,i,1))/dx) + S(j,1);
                U_(j,2) = U(j,i,2) + time_step*(J(j,i,2) - (F(j,i,2)-F(j-1,i,2))/dx) + S(j,2);
                U_(j,3) = U(j,i,3) + time_step*(J(j,i,3) - (F(j,i,3)-F(j-1,i,3))/dx) + S(j,3);
            end
        end

        % Predictor Decode
        Rho_(:) = U_(:,1);
        u_(:) = U_(:,2)./Rho_(:);
        e_(:) = U_(:,3)./Rho_(:);
        T_(:) = e_(:)./Cv;
        P_(:) = Rho_(:).*R.*T_(:);

        % Predictor Boundary Conditions
        u_(1) = 0;
        u_(end) = 0;
        T_(1) = (4/3)*T_(2) - (1/3)*T_(3);
        T_(end) = (4/3)*T_(end-1) - (1/3)*T_(end-2);
        P_(1) = (4/3)*P_(2) - (1/3)*P_(3);
        P_(end) = (4/3)*P_(end-1) - (1/3)*P_(end-2);
        Rho_(:) = P_(:)./(R.*T_(:));
        e_(:) = Cv.*T_(:);

        % Predictor Viscous Transport Properties
        Tao_ = zeros(length(x), 1);
        qx_ = zeros(length(x), 1);
        Mu_ = Mu_ref*(T_/T_ref).^(3/2).*(T_ref+110)./(T_+110);
        Lambda_ = -(2/3)*Mu_;
        K_ = Mu_*Cp/Pr;
        for j = 1:length(x)
            if j == 1
                du_j = (-3*u_(j)+4*u_(j+1)-u_(j+2))/(2*dx);
                dT_j = (-3*T_(j)+4*T_(j+1)-T_(j+2))/(2*dx);
            elseif j == length(x)
                du_j = (3*u_(j)-4*u_(j-1)+u_(j-2))/(2*dx);
                dT_j = (3*T_(j)-4*T_(j-1)+T_(j-2))/(2*dx);
            else
                du_j = (u_(j+1)-u_(j-1))/(2*dx);
                dT_j = (T_(j+1)-T_(j-1))/(2*dx);
            end
            Tao_(j) = (Lambda_(j)+2*Mu_(j))*du_j;
            qx_(j) = -K_(j)*dT_j;
        end

        % Predictor J
        for j = 1:length(x)
            if j == 1
                du = (-3*u_(j)+4*u_(j+1)-u_(j+2))/(2*dx);
                J_(j,2) = 0;
                J_(j,3) = -P_(j)*du;
            elseif j == length(x)
                du = (3*u_(j)-4*u_(j-1)+u_(j-2))/(2*dx);
                J_(j,2) = 0;
                J_(j,3) = -P_(j)*du;
            else
                du = (u_(j+1)-u_(j-1))/(2*dx);
                J_(j,2) = (Tao_(j+1)-Tao_(j-1))/(2*dx);
                J_(j,3) = -P_(j)*du + Tao_(j)*du + (qx_(j+1)-qx_(j-1))/(2*dx);
            end
        end

        % Predictor F
        F_(:,1) = Rho_(:).*u_(:);
        F_(:,2) = Rho_(:).*u_(:).^2 + P_(:);
        F_(:,3) = Rho_(:).*e_(:).*u_(:);

        U_(:,1) = Rho_(:);
        U_(:,2) = Rho_(:).*u_(:);
        U_(:,3) = Rho_(:).*e_(:);

        % Predictor Artificial Dissipation S_
        S_ = zeros(length(x), 3);
        for j = 2:length(x)-1
            coeff = Cx*abs(P_(j+1)-2*P_(j)+P_(j-1))/(P_(j+1)+2*P_(j)+P_(j-1));
            S_(j,1) = coeff*(U_(j+1,1)-2*U_(j,1)+U_(j-1,1));
            S_(j,2) = coeff*(U_(j+1,2)-2*U_(j,2)+U_(j-1,2));
            S_(j,3) = coeff*(U_(j+1,3)-2*U_(j,3)+U_(j-1,3));
        end

        % MacCormack Scheme - Corrector
        for j = 1:length(x)
            if j == 1
                U(j,i+1,1) = 0.5*(U(j,i,1)+U_(j,1) + time_step*(J_(j,1) - (-3*F_(j,1)+4*F_(j+1,1)-F_(j+2,1))/(2*dx)) + S_(j,1));
                U(j,i+1,2) = 0.5*(U(j,i,2)+U_(j,2) + time_step*(J_(j,2) - (-3*F_(j,2)+4*F_(j+1,2)-F_(j+2,2))/(2*dx)) + S_(j,2));
                U(j,i+1,3) = 0.5*(U(j,i,3)+U_(j,3) + time_step*(J_(j,3) - (-3*F_(j,3)+4*F_(j+1,3)-F_(j+2,3))/(2*dx)) + S_(j,3));
            elseif j == length(x)
                U(j,i+1,1) = 0.5*(U(j,i,1)+U_(j,1) + time_step*(J_(j,1) - (3*F_(j,1)-4*F_(j-1,1)+F_(j-2,1))/(2*dx)) + S_(j,1));
                U(j,i+1,2) = 0.5*(U(j,i,2)+U_(j,2) + time_step*(J_(j,2) - (3*F_(j,2)-4*F_(j-1,2)+F_(j-2,2))/(2*dx)) + S_(j,2));
                U(j,i+1,3) = 0.5*(U(j,i,3)+U_(j,3) + time_step*(J_(j,3) - (3*F_(j,3)-4*F_(j-1,3)+F_(j-2,3))/(2*dx)) + S_(j,3));
            else
                U(j,i+1,1) = 0.5*(U(j,i,1)+U_(j,1) + time_step*(J_(j,1) - (F_(j+1,1)-F_(j,1))/dx) + S_(j,1));
                U(j,i+1,2) = 0.5*(U(j,i,2)+U_(j,2) + time_step*(J_(j,2) - (F_(j+1,2)-F_(j,2))/dx) + S_(j,2));
                U(j,i+1,3) = 0.5*(U(j,i,3)+U_(j,3) + time_step*(J_(j,3) - (F_(j+1,3)-F_(j,3))/dx) + S_(j,3));
            end
        end

        % Corrector Decode
        Rho(:,i+1) = U(:,i+1,1);
        u(:,i+1) = U(:,i+1,2)./Rho(:,i+1);
        e(:,i+1) = U(:,i+1,3)./Rho(:,i+1);
        T(:,i+1) = e(:,i+1)./Cv;
        P(:,i+1) = Rho(:,i+1).*R.*T(:,i+1);

        % Corrector Boundary Conditions
        u(1,i+1) = 0;
        u(end,i+1) = 0;
        T(1,i+1) = (4/3)*T(2,i+1) - (1/3)*T(3,i+1);
        T(end,i+1) = (4/3)*T(end-1,i+1) - (1/3)*T(end-2,i+1);
        P(1,i+1) = (4/3)*P(2,i+1) - (1/3)*P(3,i+1);
        P(end,i+1) = (4/3)*P(end-1,i+1) - (1/3)*P(end-2,i+1);
        Rho(:,i+1) = P(:,i+1)./(R.*T(:,i+1));
        e(:,i+1) = Cv.*T(:,i+1);

    end
end

%% Analytical Solution
t_anal = 4e-3;
PR = 3.031;
alpha = (Gamma+1)/(Gamma-1);
T1_a = P1/(Rho1*R);
a1 = sqrt(Gamma*R*T1_a);
T4_a = P4/(Rho4*R);
a4 = sqrt(Gamma*R*T4_a);

P2 = PR*P1;
rho2 = Rho1*(1+alpha*PR)/(alpha+PR);
T2_a = P2/(rho2*R);
u2 = (PR-1)/sqrt((1+alpha*PR)*Gamma*(Gamma-1)*0.5)*a1;

P3 = P2;
u3 = u2;
rho3 = Rho4*(P3/P4)^(1/Gamma);
T3_a = P3/(rho3*R);
a3 = sqrt(Gamma*R*T3_a);

W = (PR-1)*a1^2/(Gamma*u2);
x_shock = xo + W*t_anal;
x_contact = xo + u2*t_anal;
x_fantail = xo + (u3-a3)*t_anal;
x_fanhead = xo - a4*t_anal;

NX_a = 1001;
xa = linspace(0, Lx, NX_a);
[ua, Pa, Ta, rhoa] = deal(zeros(1, NX_a));

for k = 1:NX_a
    xk = xa(k);
    if xk <= x_fanhead
        Pa(k) = P4;
        rhoa(k) = Rho4;
        Ta(k) = T4_a;
        ua(k) = 0;
    elseif xk <= x_fantail
        uk = (2/(Gamma+1))*(a4+(xk-xo)/t_anal);
        Tk = T4_a*(1-(Gamma-1)/2*uk/a4)^2;
        rk = Rho4*(1-(Gamma-1)/2*uk/a4)^(2/(Gamma-1));
        Pa(k) = rk*R*Tk;
        rhoa(k) = rk;
        Ta(k) = Tk;
        ua(k) = uk;
    elseif xk <= x_contact
        Pa(k) = P3;
        rhoa(k) = rho3;
        Ta(k) = T3_a;
        ua(k) = u3;
    elseif xk <= x_shock
        Pa(k) = P2;
        rhoa(k) = rho2;
        Ta(k) = T2_a;
        ua(k) = u2;
    else
        Pa(k) = P1;
        rhoa(k) = Rho1;
        Ta(k) = T1_a;
        ua(k) = 0;
    end
end

%% Task 1 Plots
figure(1)
clf
subplot(221)
plot(x, u(:,end), 'b-', 'LineWidth', 1.5)
hold on
plot(xa, ua, 'r--', 'LineWidth', 1.5)
xlabel('x (m)')
title('u (m/s)')
legend('Viscous (Numerical)', 'Analytical (Inviscid)')

subplot(222)
plot(x, P(:,end), 'b-', 'LineWidth', 1.5)
hold on
plot(xa, Pa, 'r--', 'LineWidth', 1.5)
xlabel('x (m)')
title('P (Pa)')
legend('Viscous (Numerical)', 'Analytical (Inviscid)')

subplot(223)
plot(x, T(:,end), 'b-', 'LineWidth', 1.5)
hold on
plot(xa, Ta, 'r--', 'LineWidth', 1.5)
xlabel('x (m)')
title('T (K)')
legend('Viscous (Numerical)', 'Analytical (Inviscid)')

subplot(224)
plot(x, Rho(:,end), 'b-', 'LineWidth', 1.5)
hold on
plot(xa, rhoa, 'r--', 'LineWidth', 1.5)
xlabel('x (m)')
title('\rho (kg/m^3)')
legend('Viscous (Numerical)', 'Analytical (Inviscid)')

sgtitle('Task 1: Viscous (Numerical) vs Analytical (Inviscid) at t = 4 ms')

%% Task 2

Nx_list = [21, 101, 1001];
Cx_list = [0.05, 0.1, 0.5];
colors = {'g-', 'b-', 'r-'};
res = struct();

for n = 1:length(Nx_list)
    Nx2 = Nx_list(n);
    Cx2 = Cx_list(n);

    x2 = 0:(Lx/Nx2):Lx;
    dx2 = x2(2)-x2(1);
    N2 = length(x2);

    [P2c, Rho2c, u2c, T2c, e2c] = deal(zeros(N2, 1));
    [P2n, Rho2n, u2n, T2n, e2n] = deal(zeros(N2, 1));
    [U2, F2, J2m] = deal(zeros(N2, 3));
    [Rho2_, u2_, e2_, P2_, T2_] = deal(zeros(N2, 1));
    [U2_, J2_, F2_, S2, S2_] = deal(zeros(N2, 3));

    ip2 = round(Nx2*(xo/Lx))+1;
    P2c(1:ip2) = P4;
    P2c(ip2:end) = P1;
    Rho2c(1:ip2) = Rho4;
    Rho2c(ip2:end) = Rho1;
    T2c = P2c./(R*Rho2c);
    e2c = Cv*T2c;
    T2c = smooth(T2c);
    u2c = smooth(u2c);
    P2c = smooth(P2c);
    Rho2c = smooth(Rho2c);
    e2c = smooth(e2c);

    for i = 1:length(time)

        % Viscous Transport Properties
        Tao2 = zeros(N2, 1);
        qx2 = zeros(N2, 1);
        Mu2 = Mu_ref*(T2c/T_ref).^(3/2).*(T_ref+110)./(T2c+110);
        Lambda2 = -(2/3)*Mu2;
        K2 = Mu2*Cp/Pr;
        for j = 1:N2
            if j == 1
                du_j = (-3*u2c(j)+4*u2c(j+1)-u2c(j+2))/(2*dx2);
                dT_j = (-3*T2c(j)+4*T2c(j+1)-T2c(j+2))/(2*dx2);
            elseif j == N2
                du_j = (3*u2c(j)-4*u2c(j-1)+u2c(j-2))/(2*dx2);
                dT_j = (3*T2c(j)-4*T2c(j-1)+T2c(j-2))/(2*dx2);
            else
                du_j = (u2c(j+1)-u2c(j-1))/(2*dx2);
                dT_j = (T2c(j+1)-T2c(j-1))/(2*dx2);
            end
            Tao2(j) = (Lambda2(j)+2*Mu2(j))*du_j;
            qx2(j) = -K2(j)*dT_j;
        end

        % J
        for j = 1:N2
            if j == 1
                du = (-3*u2c(j)+4*u2c(j+1)-u2c(j+2))/(2*dx2);
                J2m(j,2) = 0;
                J2m(j,3) = -P2c(j)*du;
            elseif j == N2
                du = (3*u2c(j)-4*u2c(j-1)+u2c(j-2))/(2*dx2);
                J2m(j,2) = 0;
                J2m(j,3) = -P2c(j)*du;
            else
                du = (u2c(j+1)-u2c(j-1))/(2*dx2);
                J2m(j,2) = (Tao2(j+1)-Tao2(j-1))/(2*dx2);
                J2m(j,3) = -P2c(j)*du + Tao2(j)*du + (qx2(j+1)-qx2(j-1))/(2*dx2);
            end
        end

        % U and F
        U2(:,1) = Rho2c;
        U2(:,2) = Rho2c.*u2c;
        U2(:,3) = Rho2c.*e2c;
        F2(:,1) = Rho2c.*u2c;
        F2(:,2) = Rho2c.*u2c.^2 + P2c;
        F2(:,3) = Rho2c.*e2c.*u2c;

        % Artificial Dissipation
        S2 = zeros(N2, 3);
        for j = 2:N2-1
            c = Cx2*abs(P2c(j+1)-2*P2c(j)+P2c(j-1))/(P2c(j+1)+2*P2c(j)+P2c(j-1));
            S2(j,1) = c*(U2(j+1,1)-2*U2(j,1)+U2(j-1,1));
            S2(j,2) = c*(U2(j+1,2)-2*U2(j,2)+U2(j-1,2));
            S2(j,3) = c*(U2(j+1,3)-2*U2(j,3)+U2(j-1,3));
        end

        if i ~= length(time)

            % MacCormack Scheme - Predictor
            for j = 1:N2
                if j == 1
                    U2_(j,1) = U2(j,1) + time_step*(J2m(j,1) - (-3*F2(j,1)+4*F2(j+1,1)-F2(j+2,1))/(2*dx2)) + S2(j,1);
                    U2_(j,2) = U2(j,2) + time_step*(J2m(j,2) - (-3*F2(j,2)+4*F2(j+1,2)-F2(j+2,2))/(2*dx2)) + S2(j,2);
                    U2_(j,3) = U2(j,3) + time_step*(J2m(j,3) - (-3*F2(j,3)+4*F2(j+1,3)-F2(j+2,3))/(2*dx2)) + S2(j,3);
                elseif j == N2
                    U2_(j,1) = U2(j,1) + time_step*(J2m(j,1) - (3*F2(j,1)-4*F2(j-1,1)+F2(j-2,1))/(2*dx2)) + S2(j,1);
                    U2_(j,2) = U2(j,2) + time_step*(J2m(j,2) - (3*F2(j,2)-4*F2(j-1,2)+F2(j-2,2))/(2*dx2)) + S2(j,2);
                    U2_(j,3) = U2(j,3) + time_step*(J2m(j,3) - (3*F2(j,3)-4*F2(j-1,3)+F2(j-2,3))/(2*dx2)) + S2(j,3);
                else
                    U2_(j,1) = U2(j,1) + time_step*(J2m(j,1) - (F2(j,1)-F2(j-1,1))/dx2) + S2(j,1);
                    U2_(j,2) = U2(j,2) + time_step*(J2m(j,2) - (F2(j,2)-F2(j-1,2))/dx2) + S2(j,2);
                    U2_(j,3) = U2(j,3) + time_step*(J2m(j,3) - (F2(j,3)-F2(j-1,3))/dx2) + S2(j,3);
                end
            end

            % Predictor Decode
            Rho2_ = U2_(:,1);
            u2_ = U2_(:,2)./Rho2_;
            e2_ = U2_(:,3)./Rho2_;
            T2_ = e2_./Cv;
            P2_ = Rho2_.*R.*T2_;

            % Predictor Boundary Conditions
            u2_(1) = 0;
            u2_(end) = 0;
            T2_(1) = (4/3)*T2_(2) - (1/3)*T2_(3);
            T2_(end) = (4/3)*T2_(end-1) - (1/3)*T2_(end-2);
            P2_(1) = (4/3)*P2_(2) - (1/3)*P2_(3);
            P2_(end) = (4/3)*P2_(end-1) - (1/3)*P2_(end-2);
            Rho2_ = P2_./(R.*T2_);
            e2_ = Cv.*T2_;

            % Predictor Viscous Transport Properties
            Tao2_ = zeros(N2, 1);
            qx2_ = zeros(N2, 1);
            Mu2_ = Mu_ref*(T2_/T_ref).^(3/2).*(T_ref+110)./(T2_+110);
            Lambda2_ = -(2/3)*Mu2_;
            K2_ = Mu2_*Cp/Pr;
            for j = 1:N2
                if j == 1
                    du_j = (-3*u2_(j)+4*u2_(j+1)-u2_(j+2))/(2*dx2);
                    dT_j = (-3*T2_(j)+4*T2_(j+1)-T2_(j+2))/(2*dx2);
                elseif j == N2
                    du_j = (3*u2_(j)-4*u2_(j-1)+u2_(j-2))/(2*dx2);
                    dT_j = (3*T2_(j)-4*T2_(j-1)+T2_(j-2))/(2*dx2);
                else
                    du_j = (u2_(j+1)-u2_(j-1))/(2*dx2);
                    dT_j = (T2_(j+1)-T2_(j-1))/(2*dx2);
                end
                Tao2_(j) = (Lambda2_(j)+2*Mu2_(j))*du_j;
                qx2_(j) = -K2_(j)*dT_j;
            end

            % Predictor J
            for j = 1:N2
                if j == 1
                    du = (-3*u2_(j)+4*u2_(j+1)-u2_(j+2))/(2*dx2);
                    J2_(j,2) = 0;
                    J2_(j,3) = -P2_(j)*du;
                elseif j == N2
                    du = (3*u2_(j)-4*u2_(j-1)+u2_(j-2))/(2*dx2);
                    J2_(j,2) = 0;
                    J2_(j,3) = -P2_(j)*du;
                else
                    du = (u2_(j+1)-u2_(j-1))/(2*dx2);
                    J2_(j,2) = (Tao2_(j+1)-Tao2_(j-1))/(2*dx2);
                    J2_(j,3) = -P2_(j)*du + Tao2_(j)*du + (qx2_(j+1)-qx2_(j-1))/(2*dx2);
                end
            end

            % Predictor F and U
            F2_(:,1) = Rho2_.*u2_;
            F2_(:,2) = Rho2_.*u2_.^2 + P2_;
            F2_(:,3) = Rho2_.*e2_.*u2_;
            U2_(:,1) = Rho2_;
            U2_(:,2) = Rho2_.*u2_;
            U2_(:,3) = Rho2_.*e2_;

            % Predictor Artificial Dissipation
            S2_ = zeros(N2, 3);
            for j = 2:N2-1
                c = Cx2*abs(P2_(j+1)-2*P2_(j)+P2_(j-1))/(P2_(j+1)+2*P2_(j)+P2_(j-1));
                S2_(j,1) = c*(U2_(j+1,1)-2*U2_(j,1)+U2_(j-1,1));
                S2_(j,2) = c*(U2_(j+1,2)-2*U2_(j,2)+U2_(j-1,2));
                S2_(j,3) = c*(U2_(j+1,3)-2*U2_(j,3)+U2_(j-1,3));
            end

            % MacCormack Scheme - Corrector
            for j = 1:N2
                if j == 1
                    U2(j,1) = 0.5*(U2(j,1)+U2_(j,1) + time_step*(J2_(j,1) - (-3*F2_(j,1)+4*F2_(j+1,1)-F2_(j+2,1))/(2*dx2))) + S2_(j,1);
                    U2(j,2) = 0.5*(U2(j,2)+U2_(j,2) + time_step*(J2_(j,2) - (-3*F2_(j,2)+4*F2_(j+1,2)-F2_(j+2,2))/(2*dx2))) + S2_(j,2);
                    U2(j,3) = 0.5*(U2(j,3)+U2_(j,3) + time_step*(J2_(j,3) - (-3*F2_(j,3)+4*F2_(j+1,3)-F2_(j+2,3))/(2*dx2))) + S2_(j,3);
                elseif j == N2
                    U2(j,1) = 0.5*(U2(j,1)+U2_(j,1) + time_step*(J2_(j,1) - (3*F2_(j,1)-4*F2_(j-1,1)+F2_(j-2,1))/(2*dx2))) + S2_(j,1);
                    U2(j,2) = 0.5*(U2(j,2)+U2_(j,2) + time_step*(J2_(j,2) - (3*F2_(j,2)-4*F2_(j-1,2)+F2_(j-2,2))/(2*dx2))) + S2_(j,2);
                    U2(j,3) = 0.5*(U2(j,3)+U2_(j,3) + time_step*(J2_(j,3) - (3*F2_(j,3)-4*F2_(j-1,3)+F2_(j-2,3))/(2*dx2))) + S2_(j,3);
                else
                    U2(j,1) = 0.5*(U2(j,1)+U2_(j,1) + time_step*(J2_(j,1) - (F2_(j+1,1)-F2_(j,1))/dx2)) + S2_(j,1);
                    U2(j,2) = 0.5*(U2(j,2)+U2_(j,2) + time_step*(J2_(j,2) - (F2_(j+1,2)-F2_(j,2))/dx2)) + S2_(j,2);
                    U2(j,3) = 0.5*(U2(j,3)+U2_(j,3) + time_step*(J2_(j,3) - (F2_(j+1,3)-F2_(j,3))/dx2)) + S2_(j,3);
                end
            end

            % Corrector Decode + BCs
            Rho2n = U2(:,1);
            u2n = U2(:,2)./Rho2n;
            e2n = U2(:,3)./Rho2n;
            T2n = e2n./Cv;
            P2n = Rho2n.*R.*T2n;
            u2n(1) = 0;
            u2n(end) = 0;
            T2n(1) = (4/3)*T2n(2) - (1/3)*T2n(3);
            T2n(end) = (4/3)*T2n(end-1) - (1/3)*T2n(end-2);
            P2n(1) = (4/3)*P2n(2) - (1/3)*P2n(3);
            P2n(end) = (4/3)*P2n(end-1) - (1/3)*P2n(end-2);
            Rho2n = P2n./(R.*T2n);
            e2n = Cv.*T2n;

            P2c = P2n;
            Rho2c = Rho2n;
            u2c = u2n;
            T2c = T2n;
            e2c = e2n;
        end
    end

    res(n).x = x2;
    res(n).u = u2c;
    res(n).P = P2c;
    res(n).T = T2c;
    res(n).Rho = Rho2c;
end

%% Task 2 Plots
lbls = {sprintf('Nx=21 (Cx=%.2f)', Cx_list(1)), ...
        sprintf('Nx=101 (Cx=%.2f)', Cx_list(2)), ...
        sprintf('Nx=1001 (Cx=%.2f)', Cx_list(3))};

figure(2)
clf
subplot(221)
hold on
grid on
for n = 1:3
    plot(res(n).x, res(n).u, colors{n}, 'LineWidth', 1.2, 'DisplayName', lbls{n})
end
plot(xa, ua, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Analytical')
xlabel('x (m)')
title('u (m/s)')
legend

subplot(222)
hold on
grid on
for n = 1:3
    plot(res(n).x, res(n).P, colors{n}, 'LineWidth', 1.2, 'DisplayName', lbls{n})
end
plot(xa, Pa, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Analytical')
xlabel('x (m)')
title('P (Pa)')
legend

subplot(223)
hold on
grid on
for n = 1:3
    plot(res(n).x, res(n).T, colors{n}, 'LineWidth', 1.2, 'DisplayName', lbls{n})
end
plot(xa, Ta, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Analytical')
xlabel('x (m)')
title('T (K)')
legend

subplot(224)
hold on
grid on
for n = 1:3
    plot(res(n).x, res(n).Rho, colors{n}, 'LineWidth', 1.2, 'DisplayName', lbls{n})
end
plot(xa, rhoa, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Analytical')
xlabel('x (m)')
title('\rho (kg/m^3)')
legend

sgtitle('Task 2: Grid Resolution Comparison at t = 4 ms')

%% Task 3

order_list = [2, 4];
ord_res = struct();

for oi = 1:2
    ord = order_list(oi);

    x3 = 0:(Lx/101):Lx;
    dx3 = x3(2)-x3(1);
    N3 = length(x3);

    [P3c, Rho3c, u3c, T3c, e3c] = deal(zeros(N3, 1));
    [P3n, Rho3n, u3n, T3n, e3n] = deal(zeros(N3, 1));
    [U3, F3, J3m] = deal(zeros(N3, 3));
    [Rho3_, u3_, e3_, P3_, T3_] = deal(zeros(N3, 1));
    [U3_, J3_, F3_, S3, S3_] = deal(zeros(N3, 3));

    ip3 = round(101*(xo/Lx))+1;
    P3c(1:ip3) = P4;
    P3c(ip3:end) = P1;
    Rho3c(1:ip3) = Rho4;
    Rho3c(ip3:end) = Rho1;
    T3c = P3c./(R*Rho3c);
    e3c = Cv*T3c;
    T3c = smooth(T3c);
    u3c = smooth(u3c);
    P3c = smooth(P3c);
    Rho3c = smooth(Rho3c);
    e3c = smooth(e3c);

    for i = 1:length(time)

        % Viscous Transport Properties
        Tao3 = zeros(N3, 1);
        qx3 = zeros(N3, 1);
        Mu3 = Mu_ref*(T3c/T_ref).^(3/2).*(T_ref+110)./(T3c+110);
        Lambda3 = -(2/3)*Mu3;
        K3 = Mu3*Cp/Pr;
        for j = 1:N3
            if j == 1
                du_j = (-3*u3c(j)+4*u3c(j+1)-u3c(j+2))/(2*dx3);
                dT_j = (-3*T3c(j)+4*T3c(j+1)-T3c(j+2))/(2*dx3);
            elseif j == N3
                du_j = (3*u3c(j)-4*u3c(j-1)+u3c(j-2))/(2*dx3);
                dT_j = (3*T3c(j)-4*T3c(j-1)+T3c(j-2))/(2*dx3);
            else
                du_j = (u3c(j+1)-u3c(j-1))/(2*dx3);
                dT_j = (T3c(j+1)-T3c(j-1))/(2*dx3);
            end
            Tao3(j) = (Lambda3(j)+2*Mu3(j))*du_j;
            qx3(j) = -K3(j)*dT_j;
        end

        % J
        for j = 1:N3
            if j == 1
                du = (-3*u3c(j)+4*u3c(j+1)-u3c(j+2))/(2*dx3);
                J3m(j,2) = 0;
                J3m(j,3) = -P3c(j)*du;
            elseif j == N3
                du = (3*u3c(j)-4*u3c(j-1)+u3c(j-2))/(2*dx3);
                J3m(j,2) = 0;
                J3m(j,3) = -P3c(j)*du;
            else
                du = (u3c(j+1)-u3c(j-1))/(2*dx3);
                J3m(j,2) = (Tao3(j+1)-Tao3(j-1))/(2*dx3);
                J3m(j,3) = -P3c(j)*du + Tao3(j)*du + (qx3(j+1)-qx3(j-1))/(2*dx3);
            end
        end

        % U and F
        U3(:,1) = Rho3c;
        U3(:,2) = Rho3c.*u3c;
        U3(:,3) = Rho3c.*e3c;
        F3(:,1) = Rho3c.*u3c;
        F3(:,2) = Rho3c.*u3c.^2 + P3c;
        F3(:,3) = Rho3c.*e3c.*u3c;

        % Artificial Dissipation
        S3 = zeros(N3, 3);
        for j = 2:N3-1
            c = Cx*abs(P3c(j+1)-2*P3c(j)+P3c(j-1))/(P3c(j+1)+2*P3c(j)+P3c(j-1));
            S3(j,1) = c*(U3(j+1,1)-2*U3(j,1)+U3(j-1,1));
            S3(j,2) = c*(U3(j+1,2)-2*U3(j,2)+U3(j-1,2));
            S3(j,3) = c*(U3(j+1,3)-2*U3(j,3)+U3(j-1,3));
        end

        if i ~= length(time)

            % MacCormack Scheme - Predictor
            for j = 1:N3
                for k = 1:3
                    if j == 1
                        dFdx = (-3*F3(j,k)+4*F3(j+1,k)-F3(j+2,k))/(2*dx3);
                    elseif j == N3
                        dFdx = (3*F3(j,k)-4*F3(j-1,k)+F3(j-2,k))/(2*dx3);
                    elseif j == 2 || ord == 2
                        dFdx = (F3(j,k)-F3(j-1,k))/dx3;
                    else
                        dFdx = (7*F3(j,k)-8*F3(j-1,k)+F3(j-2,k))/(6*dx3);
                    end
                    U3_(j,k) = U3(j,k) + time_step*(J3m(j,k) - dFdx) + S3(j,k);
                end
            end

            % Predictor Decode
            Rho3_ = U3_(:,1);
            u3_ = U3_(:,2)./Rho3_;
            e3_ = U3_(:,3)./Rho3_;
            T3_ = e3_./Cv;
            P3_ = Rho3_.*R.*T3_;

            % Predictor Boundary Conditions
            u3_(1) = 0;
            u3_(end) = 0;
            T3_(1) = (4/3)*T3_(2) - (1/3)*T3_(3);
            T3_(end) = (4/3)*T3_(end-1) - (1/3)*T3_(end-2);
            P3_(1) = (4/3)*P3_(2) - (1/3)*P3_(3);
            P3_(end) = (4/3)*P3_(end-1) - (1/3)*P3_(end-2);
            Rho3_ = P3_./(R.*T3_);
            e3_ = Cv.*T3_;

            % Predictor Viscous Transport Properties
            Tao3_ = zeros(N3, 1);
            qx3_ = zeros(N3, 1);
            Mu3_ = Mu_ref*(T3_/T_ref).^(3/2).*(T_ref+110)./(T3_+110);
            Lambda3_ = -(2/3)*Mu3_;
            K3_ = Mu3_*Cp/Pr;
            for j = 1:N3
                if j == 1
                    du_j = (-3*u3_(j)+4*u3_(j+1)-u3_(j+2))/(2*dx3);
                    dT_j = (-3*T3_(j)+4*T3_(j+1)-T3_(j+2))/(2*dx3);
                elseif j == N3
                    du_j = (3*u3_(j)-4*u3_(j-1)+u3_(j-2))/(2*dx3);
                    dT_j = (3*T3_(j)-4*T3_(j-1)+T3_(j-2))/(2*dx3);
                else
                    du_j = (u3_(j+1)-u3_(j-1))/(2*dx3);
                    dT_j = (T3_(j+1)-T3_(j-1))/(2*dx3);
                end
                Tao3_(j) = (Lambda3_(j)+2*Mu3_(j))*du_j;
                qx3_(j) = -K3_(j)*dT_j;
            end

            % Predictor J
            for j = 1:N3
                if j == 1
                    du = (-3*u3_(j)+4*u3_(j+1)-u3_(j+2))/(2*dx3);
                    J3_(j,2) = 0;
                    J3_(j,3) = -P3_(j)*du;
                elseif j == N3
                    du = (3*u3_(j)-4*u3_(j-1)+u3_(j-2))/(2*dx3);
                    J3_(j,2) = 0;
                    J3_(j,3) = -P3_(j)*du;
                else
                    du = (u3_(j+1)-u3_(j-1))/(2*dx3);
                    J3_(j,2) = (Tao3_(j+1)-Tao3_(j-1))/(2*dx3);
                    J3_(j,3) = -P3_(j)*du + Tao3_(j)*du + (qx3_(j+1)-qx3_(j-1))/(2*dx3);
                end
            end

            % Predictor F and U
            F3_(:,1) = Rho3_.*u3_;
            F3_(:,2) = Rho3_.*u3_.^2 + P3_;
            F3_(:,3) = Rho3_.*e3_.*u3_;
            U3_(:,1) = Rho3_;
            U3_(:,2) = Rho3_.*u3_;
            U3_(:,3) = Rho3_.*e3_;

            % Predictor Artificial Dissipation
            S3_ = zeros(N3, 3);
            for j = 2:N3-1
                c = Cx*abs(P3_(j+1)-2*P3_(j)+P3_(j-1))/(P3_(j+1)+2*P3_(j)+P3_(j-1));
                S3_(j,1) = c*(U3_(j+1,1)-2*U3_(j,1)+U3_(j-1,1));
                S3_(j,2) = c*(U3_(j+1,2)-2*U3_(j,2)+U3_(j-1,2));
                S3_(j,3) = c*(U3_(j+1,3)-2*U3_(j,3)+U3_(j-1,3));
            end

            % MacCormack Scheme - Corrector
            % U^{t+dt}_i = 0.5*(U^t_i + U~_i + dt*(J~_i - dF/dx_forward)) + S~_i
            for j = 1:N3
                for k = 1:3
                    if j == 1
                        dFdx = (-3*F3_(j,k)+4*F3_(j+1,k)-F3_(j+2,k))/(2*dx3);
                    elseif j == N3
                        dFdx = (3*F3_(j,k)-4*F3_(j-1,k)+F3_(j-2,k))/(2*dx3);
                    elseif j == N3-1 || ord == 2
                        dFdx = (F3_(j+1,k)-F3_(j,k))/dx3;
                    else
                        dFdx = (-7*F3_(j,k)+8*F3_(j+1,k)-F3_(j+2,k))/(6*dx3);
                    end
                    U3(j,k) = 0.5*(U3(j,k)+U3_(j,k) + time_step*(J3_(j,k) - dFdx)) + S3_(j,k);
                end
            end

            % Corrector Decode + BCs
            Rho3n = U3(:,1);
            u3n = U3(:,2)./Rho3n;
            e3n = U3(:,3)./Rho3n;
            T3n = e3n./Cv;
            P3n = Rho3n.*R.*T3n;
            u3n(1) = 0;
            u3n(end) = 0;
            T3n(1) = (4/3)*T3n(2) - (1/3)*T3n(3);
            T3n(end) = (4/3)*T3n(end-1) - (1/3)*T3n(end-2);
            P3n(1) = (4/3)*P3n(2) - (1/3)*P3n(3);
            P3n(end) = (4/3)*P3n(end-1) - (1/3)*P3n(end-2);
            Rho3n = P3n./(R.*T3n);
            e3n = Cv.*T3n;

            P3c = P3n;
            Rho3c = Rho3n;
            u3c = u3n;
            T3c = T3n;
            e3c = e3n;
        end
    end

    ord_res(oi).x = x3;
    ord_res(oi).u = u3c;
    ord_res(oi).P = P3c;
    ord_res(oi).T = T3c;
    ord_res(oi).Rho = Rho3c;
end

%% Task 3 Plots
figure(3)
clf
subplot(221)
hold on
grid on
plot(ord_res(1).x, ord_res(1).u, 'b-', 'LineWidth', 1.5, 'DisplayName', '2nd-Order')
plot(ord_res(2).x, ord_res(2).u, 'r-', 'LineWidth', 1.5, 'DisplayName', '4th-Order')
plot(xa, ua, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Analytical')
xlabel('x (m)')
title('u (m/s)')
legend

subplot(222)
hold on
grid on
plot(ord_res(1).x, ord_res(1).P, 'b-', 'LineWidth', 1.5, 'DisplayName', '2nd-Order')
plot(ord_res(2).x, ord_res(2).P, 'r-', 'LineWidth', 1.5, 'DisplayName', '4th-Order')
plot(xa, Pa, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Analytical')
xlabel('x (m)')
title('P (Pa)')
legend

subplot(223)
hold on
grid on
plot(ord_res(1).x, ord_res(1).T, 'b-', 'LineWidth', 1.5, 'DisplayName', '2nd-Order')
plot(ord_res(2).x, ord_res(2).T, 'r-', 'LineWidth', 1.5, 'DisplayName', '4th-Order')
plot(xa, Ta, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Analytical')
xlabel('x (m)')
title('T (K)')
legend

subplot(224)
hold on
grid on
plot(ord_res(1).x, ord_res(1).Rho, 'b-', 'LineWidth', 1.5, 'DisplayName', '2nd-Order')
plot(ord_res(2).x, ord_res(2).Rho, 'r-', 'LineWidth', 1.5, 'DisplayName', '4th-Order')
plot(xa, rhoa, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Analytical')
xlabel('x (m)')
title('\rho (kg/m^3)')
legend

sgtitle('Task 3: 2nd-Order vs 4th-Order Scheme, Nx=101, t = 4 ms')

%% Report Requirement 1

t_snap = [1e-3, 2e-3, 4e-3];
snap_cols = round(t_snap/time_step) + 1;

figure(4)
clf
subplot(221)
hold on
grid on
for s = 1:length(snap_cols)
    plot(x, u(:,snap_cols(s)), 'DisplayName', sprintf('t = %.0f ms', t_snap(s)*1e3))
end
xlabel('x (m)')
ylabel('u (m/s)')
title('u (m/s)')
legend

subplot(222)
hold on
grid on
for s = 1:length(snap_cols)
    plot(x, P(:,snap_cols(s)), 'DisplayName', sprintf('t = %.0f ms', t_snap(s)*1e3))
end
xlabel('x (m)')
ylabel('P (Pa)')
title('P (Pa)')
legend

subplot(223)
hold on
grid on
for s = 1:length(snap_cols)
    plot(x, T(:,snap_cols(s)), 'DisplayName', sprintf('t = %.0f ms', t_snap(s)*1e3))
end
xlabel('x (m)')
ylabel('T (K)')
title('T (K)')
legend

subplot(224)
hold on
grid on
for s = 1:length(snap_cols)
    plot(x, Rho(:,snap_cols(s)), 'DisplayName', sprintf('t = %.0f ms', t_snap(s)*1e3))
end
xlabel('x (m)')
ylabel('\rho (kg/m^3)')
title('\rho (kg/m^3)')
legend

sgtitle('P, \rho, u, T vs x at t = 1, 2, 4 ms')

%% Smooth Function
function T2 = smooth(T)
T2 = zeros(size(T));
T3 = zeros(size(T));
T2(2:end) = 0.5*(T(2:end)+T(1:end-1));
T2(1) = T(1);
T3(3:end) = 0.5*(T2(3:end)+T2(2:end-1));
T3(1:2) = T(1:2);
T2 = T3;
end
