select  count(*) from votes as v,          badges as b,         users as u where u.Id = v.UserId 	and v.UserId = b.UserId  AND u.Reputation>=1  AND u.Views>=0;
