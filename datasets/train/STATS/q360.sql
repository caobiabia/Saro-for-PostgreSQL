select  count(*) from comments as c,          postHistory as ph,          users as u where u.Id = c.UserId 	and c.UserId = ph.UserId  AND u.Reputation<=1226  AND u.Views>=0  AND u.DownVotes>=0;
